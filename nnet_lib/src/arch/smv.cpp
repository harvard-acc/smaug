#include <math.h>
#include <string.h>

#include "gem5/m5ops.h"

#include "nnet_fwd.h"
#include "core/ref/activation_functions.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
#include "core/smv/smv.h"
#include "utility/compression.h"
#include "utility/data_layout_conversion.h"
#include "utility/profiling.h"
#include "utility/utility.h"
#include "arch/common.h"
#include "arch/interface.h"
#include "arch/smiv/common.h"
#include "arch/smv/common.h"

#ifdef __cplusplus
#include "mkldnn.hpp"
#include "arch/nnet_mkl.h"
#include "core/mkl/activation_functions.h"
#include "core/mkl/batch_norm.h"
#include "core/mkl/pooling.h"
#include "utility/mkl/utility.h"
#endif

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == SMV

smv_global g_smv;

void init_smv_global() {
    // Use the same accelerator id for all hardware blocks. This means we will
    // simulate only ONE datapath instead of multiple, which means that the two
    // blocks can share the scratchpads (without any infrastructure
    // changes). The key is that we still trace the functions at the _hw level,
    // so that Aladdin will exit after simulating each block, and we can return
    // control to the CPU at the right places.  In contrast, if we used two
    // different ids, we would have two different datapaths that could not share
    // data directly.
    g_smv.kConvolutionHw = 0x0003;
    g_smv.kInnerProductHw = 0x0003;
    g_smv.kReductionHw = 0x0003;
    g_smv.kBatchNormHw = 0x0003;
    g_smv.kPoolingHw = 0x0003;

    g_smv.umem = (float*)malloc_aligned(SMV_UMEM_SIZE);
    g_smv.spad0 = (float*)malloc_aligned(SMV_SPAD_SIZE);
    g_smv.spad1 = (float*)malloc_aligned(SMV_SPAD_SIZE);
}

void free_smv_global() {
    free(g_smv.umem);
    free(g_smv.spad0);
    free(g_smv.spad1);
}

result_buf smv_activation_function(float* activations,
                                    layer_t* layer,
                                    float* results,
                                    device_t* device) {
    // MKL seems to have particularly poor performing activation function
    // implementations.
#if 0
    begin_ignored_profiling(layer->num);
    nnet_mkl::activation_fun(
            activations, NUM_TEST_CASES, layer, results, device);
    end_profiling();
    nnet_mkl::MklSession* session = nnet_mkl::get_session(device);
    session->run_and_clear();
    return results;
#else
    int output_size = get_dims_size(&layer->outputs);
    begin_profiling(ACTIVATION_TYPE_STR(layer->activation), layer->num);
    activation_fun(activations, NUM_TEST_CASES, output_size, layer->activation);
    end_profiling();
    return activations;
#endif
}

result_buf flatten_input(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    begin_profiling(__func__, lnum);
    result_buf result_loc = im2row(activations, layers, lnum, result);
    end_profiling();
    return result_loc;
}

result_buf inner_product_layer(float* host_activations,
                               float* host_weights,
                               layer_t* layers,
                               int lnum,
                               float* host_result,
                               device_t* device,
                               sampling_param_t* sampling_param) {
    smv_inner_product_layer_impl(host_activations, host_weights, layers, lnum,
                                 host_result, &g_smv, device);
    return host_result;
}

result_buf standard_convolution_layer(float* activations,
                                      float* weights,
                                      layer_t* layers,
                                      int lnum,
                                      float* result,
                                      device_t* device,
                                      sampling_param_t* sampling_param) {
    // TODO: Consider pipelining activation function or pooling layer with
    // convolution while the accelerator is running! This may require the use
    // of pthreads (and the memory management could get messy too...), but it
    // would get us more performance.
    layer_t curr_layer = layers[lnum];
    packed_fp16* packed_weights =
            curr_layer.host_weights.data[0].dense_hp->d;
    smv_standard_convolution_layer_impl(activations,
                                        packed_weights,
                                        layers,
                                        lnum,
                                        result,
                                        &g_smv,
                                        device,
                                        sampling_param);
    return result;
}

// TODO: Depthwise convolutions fall back on SMIV.
result_buf depthwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* result,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    float* current_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);
    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        copy_zeropad(activations, layers, lnum, result);
        PRINT_MSG("After zeropadding:\n");
        PRINT_DEBUG4D(result,
                      curr_layer.inputs.rows,
                      curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                      curr_layer.inputs.height);
        PRINT_DEBUG4D_V(weights, curr_layer.weights.rows,
                        curr_layer.weights.cols + curr_layer.weights.align_pad,
                        curr_layer.weights.height);
        smiv_depthwise_convolution_layer_impl(result, current_layer_weights,
                                              layers, lnum, activations,
                                              (smiv_global*)&g_smv, device);

        return activations;
    }
    smiv_depthwise_convolution_layer_impl(activations, current_layer_weights,
                                          layers, lnum, result,
                                          (smiv_global*)&g_smv, device);

    return result;
}

// TODO: This falls back to SMIV's implementation because we know it works.
// But SMV's inner product routine is much faster. Use it instead.
result_buf pointwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    // Allocate memory to store the transformed input.
    float* nhwc_inputs = NULL;
    dims_t nhwc = convert_nchw_to_nhwc(activations, NUM_TEST_CASES,
                                       layers[lnum].inputs, DATA_ALIGNMENT,
                                       &nhwc_inputs);

    // HACK: We need to modify the layer[lnum] descriptor to reflect the fact
    // that we're doing a matrix multiply, but these changes can't be seen
    // outside of this function. So, back up the current inputs layer.
    layer_t old_layer = layers[lnum];
    // These are the dimensions needed by the FC block routine.
    dims_t fc_dims = { nhwc.height * nhwc.rows, nhwc.cols, 1, nhwc.align_pad };
    layers[lnum].inputs = fc_dims;

    // These are the outputs dimensions expected by the FC block.
    int weights_cols = layers[lnum].weights.cols;
    layers[lnum].outputs =
            (dims_t){ fc_dims.rows, weights_cols, 1,
                      calc_padding(weights_cols, DATA_ALIGNMENT) };

    // Allocate new memory to store the result of the FC. The
    // activations/results buffers are not necessarily big enough to store this
    // (due to data alignment).
    float* nhwc_outputs = (float*)malloc_aligned(
            get_dims_size(&layers[lnum].outputs) * sizeof(float));

    // Finally, invoke the FC hardware.
    smiv_inner_product_layer_impl(nhwc_inputs, weights, layers, lnum,
                                  nhwc_outputs, (smiv_global*)&g_smv, device);

    PRINT_MSG_V("1x1 GEMM results:\n");
    PRINT_DEBUG_V(nhwc_outputs, fc_dims.rows,
                  layers[lnum].weights.cols + layers[lnum].weights.align_pad,
                  layers[lnum].weights.cols + layers[lnum].weights.align_pad);

    // Reshape the FC results and convert back from NHWC to NCHW.
    dims_t output_dims = {
        old_layer.outputs.cols,
        old_layer.outputs.height,
        old_layer.outputs.rows,
        calc_padding(old_layer.outputs.height, DATA_ALIGNMENT)
    };
    convert_nhwc_to_nchw(nhwc_outputs, NUM_TEST_CASES, output_dims,
                         DATA_ALIGNMENT, &results);

    // Restore the original layer descriptor.
    layers[lnum] = old_layer;

    free(nhwc_inputs);
    free(nhwc_outputs);
    return results;
}

// TODO: Fallback on SMIV.
result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result,
                         device_t* device,
                         sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    if (device->use_hw_pooling) {
        smv_pooling_layer_impl(
                activations, &layers[lnum], &g_smv, result, device);
    } else {
#ifdef __cplusplus
        if (curr_layer.pool == MAX) {
            nnet_mkl::max_pooling_3d(
                    activations, &layers[lnum], result, device);
        } else if (curr_layer.pool == AVG) {
            nnet_mkl::avg_pooling_3d(
                    activations, &layers[lnum], result, device);
        } else {
            assert(false && "Unsupported pooling layer type!");
        }
        nnet_mkl::MklSession* session = nnet_mkl::get_session(device);
        session->run_and_clear();
#else
        // This code should only get run by the tracer.
        if (curr_layer.pool == MAX) {
            max_pooling(activations, result, layers[lnum]);
        } else if (curr_layer.pool == AVG) {
            avg_pooling(activations, result, layers[lnum]);
        } else {
            assert(false && "Unsupported pooling layer type!");
        }
#endif
    }
    return result;
}

result_buf batch_norm_layer(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    smv_batch_norm_layer_impl(
            activations, weights, layers, lnum, result, &g_smv, device);
    return result;
}


result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     device_t* device,
                     sampling_param_t* sampling_param) {
    begin_profiling("run_layer", layer_num);

    begin_profiling("layer_dispatcher", layer_num);
    result_buf result_loc = layer_dispatcher(activations,
                                             weights,
                                             layers,
                                             layer_num,
                                             result,
                                             device,
                                             sampling_param);
    end_profiling();

    // SMV supports the same activation functions as SMIV (for now).
    activation_type act_func = layers[layer_num].activation;
    bool do_activation = act_func != NO_ACTIVATION;
    bool do_hw_activation =
            device->use_hw_activation_func &&
            smiv_is_supported_activation_func(layers[layer_num].type, act_func);
    bool use_pipelined_activation = device->use_pipelined_activation_func;
    if (do_activation && !do_hw_activation && !use_pipelined_activation) {
        if (result_loc == activations) {
            result_loc = smv_activation_function(
                    activations, &layers[layer_num], result, device);
        } else {
            result_loc = smv_activation_function(
                    result, &layers[layer_num], activations, device);
        }
        PRINT_MSG("\nactivation function\n");
        PRINT_DEBUG4D(result_loc, layers[layer_num].outputs.rows,
                      layers[layer_num].outputs.cols +
                              layers[layer_num].outputs.align_pad,
                      layers[layer_num].outputs.height);
    }
    end_profiling();
    return result_loc;
}

// Set the IO required flags for each layer.
//
// Since SMV can share scratchpads between the conv/fc blocks, we only need
// IO if we need to send data back to the CPU.
void set_io_requirements(network_t* network, device_t* device) {
    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        layer_t* curr_layer = &network->layers[layer_num];

        // The input layer is easy.
        if (layer_num == 0) {
            curr_layer->input_req = IO_NONE;
            curr_layer->output_req = device->cpu_default_offload;
            continue;
        }
        // All weights, by default, must be copied, unless the layer is an FC
        // using compressed weights (in which case the compression HW does the
        // copy).
        curr_layer->weights_req = device->cpu_default_offload;

        layer_t* prev_layer = &network->layers[layer_num - 1];
        layer_t* next_layer = &network->layers[layer_num + 1];
#if DEBUG_LEVEL > 0
        // When debugging, if we don't send the results back, we won't be able
        // to see what's happening.
        curr_layer->input_req = device->cpu_default_offload;
        curr_layer->output_req = device->cpu_default_offload;
#else

        // First, determine if we need to send the output back or if we can
        // cache it in the scratchpads locally.
        if (layer_num == network->depth - 1 ||
            // All these activation functions are unsupported.
            curr_layer->activation == SOFTMAX ||
            // If we disabled HW activation functions but an activation
            // function is necessary, we need to send back results.
            (!device->use_hw_activation_func &&
             curr_layer->activation != NO_ACTIVATION) ||
            // For now, conv layers also do not support local caching.
            curr_layer->type == CONV_STANDARD ||
            curr_layer->type == CONV_DEPTHWISE ||
            curr_layer->type == CONV_POINTWISE ||
            // We need to do data layout on the CPU before invoking pooling
            // block and we don't support local caching for batch norm right
            // now.
            curr_layer->type == BATCH_NORM ||
            next_layer->type == BATCH_NORM ||
            curr_layer->type == POOLING ||
            next_layer->type == POOLING ||
            // If the FC block needs work division, we can't locally cache.
            (curr_layer->type == FC && next_layer->type == FC &&
             smv_inner_product_needs_work_division(
                     &network->layers[layer_num]))) {
            curr_layer->output_req = device->cpu_default_offload;
        } else {
            curr_layer->output_req = IO_NONE;
        }
        // We also support one particular case where we only use ACP/CACHE for
        // results, but weights/activations will still be transferred by DMA.
        if (device->cpu_activation_func_offload != device->cpu_default_offload) {
            if ((device->cpu_activation_func_offload == IO_ACP ||
                 device->cpu_activation_func_offload == IO_CACHE) &&
                device->cpu_default_offload == IO_DMA) {
                curr_layer->output_req = device->cpu_activation_func_offload;
            } else {
                printf("[ERROR]: If cpu_activation_func_offload != "
                       "cpu_default_offload, then cpu_default_offload must be "
                       "IO_DMA.\n");
                assert(false);
            }
        }
        // We only do flattening on the CPU, so if the current layer needs
        // flattening, the previous layer needs to send results back to the
        // CPU, which means it cannot be locally cached.
        if (curr_layer->input_preprocessing == FLATTEN) {
            if (prev_layer->output_req == IO_NONE)
                prev_layer->output_req = prev_layer->input_req;
        }
        // If the previous layer doesn't need to send back results (e.g.,
        // FC->FC caching), the current layer needs no IO for inputs.
        if (prev_layer->output_req == IO_NONE) {
            curr_layer->input_req = IO_NONE;
        } else {
            curr_layer->input_req = prev_layer->output_req;
        }
#endif
    }

    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        printf("Layer %d: dmaLoad = %d, dmaStore = %d\n", layer_num,
               network->layers[layer_num].input_req,
               network->layers[layer_num].output_req);
    }

}

// Perform SMV-specific weight conversion tasks before running the network.
//
// The specific conversion task depends on the layer type:
//   1. Convolution: convert NCHW to NHWC, quantize to 16 bits.
//   2. Batch norm: quantize to 16 bits.
//   3. Fully connected: quantize to 16 bits.
//
// Converted weight descriptors are stored in the host_weights list of the
// layer descriptor. The existing weights_data structure is replaced by the new
// one created here, and its memory freed.
void early_convert_weights_data_layout(network_t* network) {
    for (int i = 1; i < network->depth; i++) {
        layer_t* layer = &network->layers[i];
        if (layer->type == CONV_STANDARD) {
            assert(layer->host_weights.len == 1 &&
                   "Standard convolutional layer must have exactly one set of "
                   "weights!");
            farray_t* nchw_weights = layer->host_weights.data[0].dense;
            farray_t nhwc_weights;
            nhwc_weights.d = NULL;
            dims_t weights_nhwc = convert_nchw_to_nhwc(
                    nchw_weights->d, layer->outputs.height, layer->weights,
                    DATA_ALIGNMENT, &nhwc_weights.d);
            nhwc_weights.size =
                    layer->outputs.height * get_dims_size(&weights_nhwc);
            uarray_t* packed_weights = pack_data_fp16(&nhwc_weights);
            layer->host_weights.data[0].dense_hp = packed_weights;
            layer->host_weights.type[0] = UncompressedHalfPrecision;
            free(nhwc_weights.d);
            free(nchw_weights);
        } else if (layer->type == BATCH_NORM) {
            assert(layer->host_weights.len == 1 &&
                   "Batch norm must have exactly one set of weights!");
            farray_t* bn_weights = layer->host_weights.data[0].dense;
            uarray_t* packed_weights = pack_data_fp16(bn_weights);
            layer->host_weights.data[0].dense_hp = packed_weights;
            layer->host_weights.type[0] = UncompressedHalfPrecision;
            free(bn_weights);
        } else if (layer->type == FC) {
            if (layer->host_weights.type[0] != Uncompressed)
                continue;  // Skip the biases.
            farray_t* weights = layer->host_weights.data[0].dense;
            uarray_t* packed_weights = pack_data_fp16(weights);
            layer->host_weights.data[0].dense_hp = packed_weights;
            layer->host_weights.type[0] = UncompressedHalfPrecision;
            free(weights);
        }
    }
}

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, activations and result.
void nnet_fwd(farray_t activations,
              farray_t weights,
              farray_t result,
              network_t network,
              device_t* device,
              sampling_param_t* sampling_param) {
    int l;
    layer_t curr_layer;

    init_smv_global();

#ifdef __cplusplus
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device->session = (void*)session;
#endif

    // Alternate between reading from/writing to activations and result so we
    // can avoid copying matrices. The initial activations is obviously in
    // "activations", so that's where we start.
    result_buf result_loc = activations.d;

    if (PRINT_DATA_AND_WEIGHTS) {
        print_data_and_weights(activations.d, weights.d, network.layers[0]);
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;

    set_io_requirements(&network, device);
    early_convert_weights_data_layout(&network);

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 0; l < network.depth; l++) {
        curr_layer = network.layers[l];

        if (result_loc == result.d) {
            result_loc = run_layer(result.d, weights.d, network.layers, l,
                                   activations.d, device, sampling_param);
        } else {
            result_loc = run_layer(activations.d, weights.d, network.layers, l,
                                   result.d, device, sampling_param);
        }
    }

    network.layers[network.depth - 1].result_in_temp = (result_loc == result.d);

#ifdef __cplusplus
    delete session;
#endif

    free_smv_global();
}

#endif
