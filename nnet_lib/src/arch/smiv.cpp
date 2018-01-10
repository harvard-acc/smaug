#include <assert.h>
#include <math.h>
#include <string.h>

#include "gem5/m5ops.h"

#include "nnet_fwd.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/ref/convolution.h"
#include "core/ref/matrix_multiply.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
#include "core/smiv/smiv.h"
#include "utility/data_layout_conversion.h"
#include "utility/profiling.h"
#include "utility/utility.h"
#include "arch/common.h"
#include "arch/interface.h"
#include "arch/smiv_common.h"

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

#if ARCHITECTURE == SMIV

// These are GLOBAL arrays which cannot be referenced directly by a HW
// function. Instead, pass them to the top level functions as function
// arguments, and use a boolean flag to indicate which one contains the data
// needed.
float* g_umem;
float* g_spad0;
float* g_spad1;

void init_work_cfg(work_cfg_t* cfg, unsigned num_iterations) {
    cfg->num_iterations = num_iterations;
    cfg->iteration = (dims_t*)malloc(sizeof(dims_t) * num_iterations);
}

void free_work_cfg(work_cfg_t* cfg) {
    free(cfg->iteration);
}

void print_work_cfg(work_cfg_t* cfg) {
    for (unsigned i = 0; i < cfg->num_iterations; i++) {
        printf("Iteration %d: height=%d, rows=%d, cols=%d, pad=%d\n",
               i,
               cfg->iteration[i].height,
               cfg->iteration[i].rows,
               cfg->iteration[i].cols,
               cfg->iteration[i].align_pad);
    }
}

// Use the same accelerator id for both the convolutional and FC blocks. This
// means we will simulate only ONE datapath instead of two, which means that
// the two blocks can share the scratchpads (without any more infrastructure
// changes). The key is that we still trace the functions at the _hw level, so
// that Aladdin will exit after simulating each block, and we can return
// control to the CPU at the right places.  In contrast, if we used two
// different ids, we would have two different datapaths that could not share
// data directly.
unsigned kConvolutionHw = 0x0003;
unsigned kInnerProductHw = 0x0003;
unsigned kReductionHw = 0x0003;
unsigned kBatchNormHw = 0x0003;

result_buf flatten_input(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    return im2row(activations, layers, lnum, result);
}

bool is_supported_activation_func(layer_type ltype, activation_type func) {
    if (ltype == FC || ltype == CONV_STANDARD || ltype == CONV_POINTWISE) {
        switch (func) {
            case NO_ACTIVATION:
            case RELU:
            case RELU_THRESHOLD:
                return true;
            default:
                return false;
        }
    } else {
        return false;
    }
}

result_buf inner_product_layer(float* host_activations,
                               float* host_weights,
                               layer_t* layers,
                               int lnum,
                               float* host_result,
                               device_t* device,
                               sampling_param_t* sampling_param) {
    inner_product_layer_impl(
            host_activations, host_weights, layers, lnum, host_result, device);
    return host_result;
}

result_buf standard_convolution_layer(float* activations,
                                      float* weights,
                                      layer_t* layers,
                                      int lnum,
                                      float* result,
                                      device_t* device,
                                      sampling_param_t* sampling_param) {

    float* current_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);
    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_weights", current_layer_weights,
                       get_num_weights_layer(layers, lnum) * sizeof(float));
    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        // TODO: Replace this with a memcpy implementation.
        copy_zeropad(activations, layers, lnum, result);
        PRINT_MSG("After zeropadding:\n");
        PRINT_DEBUG4D(result,
                      curr_layer.inputs.rows,
                      curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                      curr_layer.inputs.height);
        PRINT_DEBUG4D_V(weights, curr_layer.weights.rows,
                        curr_layer.weights.cols + curr_layer.weights.align_pad,
                        curr_layer.weights.height);
        standard_convolution_layer_impl(result,
                                        current_layer_weights,
                                        layers,
                                        lnum,
                                        activations,
                                        device,
                                        sampling_param);
        return activations;
    }
    standard_convolution_layer_impl(activations,
                                    current_layer_weights,
                                    layers,
                                    lnum,
                                    result,
                                    device,
                                    sampling_param);
    return result;
}

result_buf depthwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* result,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    float* current_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);
    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_weights", current_layer_weights,
                       get_num_weights_layer(layers, lnum) * sizeof(float));
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
        depthwise_convolution_layer_impl(result, current_layer_weights, layers,
                                         lnum, activations, device);

        return activations;
    }
    depthwise_convolution_layer_impl(
            activations, current_layer_weights, layers, lnum, result, device);

    return result;
}

// SMIV currently uses the FC block to implement a GEMM-based 1x1 convolution
// because the current model of SMIV in Aladdin doesn't support 1x1 conv on the
// CONV block.  Eventually we'll want to use the CNN block since the CNN block
// outputs results in NCHW format (where as the FC block outputs data in NHWC
// format).
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
    inner_product_layer_impl(
            nhwc_inputs, weights, layers, lnum, nhwc_outputs, device);

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

// Software implementation. SMIV doesn't accelerate pooling.
result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result,
                         device_t* device,
                         sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
#ifdef __cplusplus
    if (curr_layer.pool == MAX) {
        nnet_mkl::max_pooling_3d(activations, &layers[lnum], result, device);
    } else if (curr_layer.pool == AVG) {
        nnet_mkl::avg_pooling_3d(activations, &layers[lnum], result, device);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    nnet_mkl::MklSession* session = nnet_mkl::get_session(device);
    session->run_and_clear();
#else
    if (curr_layer.pool == MAX) {
        max_pooling(activations, result, layers[lnum]);
    } else if (curr_layer.pool == AVG) {
        avg_pooling(activations, result, layers[lnum]);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
#endif
    return result;
}

void batch_norm_layer_hw(float* host_activations,
                         float* host_weights,
                         float* host_result,
                         float* umem,
                         float* spad0,
                         float* spad1,
                         layer_t* curr_layer) {
    // DMA in the weights (to UMEM)
    setReadyBits(umem, UMEM_SIZE, 0);
    dmaLoad(umem, host_weights, WEIGHT_BYTES(curr_layer, 0));

    // DMA in the inputs (to SPAD0)
    if (curr_layer->input_req == IO_DMA) {
        grab_input_activations_dma(host_activations, spad0, curr_layer);
    }

    // The main kernel
    batch_norm_fxp(spad0, umem, curr_layer, NUM_TEST_CASES, spad1);

    // DMA out the result (from SPAD1)
    if (curr_layer->output_req == IO_DMA) {
        store_output_activations_dma(host_result, spad1, curr_layer);
    }
}

result_buf batch_norm_layer(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    float* curr_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);

    if (device->use_hw_batch_norm) {
        int weights_size = WEIGHT_BYTES(layers, lnum);
        if (weights_size > UMEM_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm weights are larger than the "
                            "UMEM - not currently supported!\n");
        }
        assert(weights_size <= UMEM_SIZE);
        int inputs_size = INPUT_BYTES(layers, lnum);
        int outputs_size = OUTPUT_BYTES(layers, lnum);
        assert(inputs_size == outputs_size);
        if (inputs_size > SPAD_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm inputs don't fit on the "
                            "scratchpad!\n");
        }
        assert(inputs_size <= SPAD_SIZE);
        MAP_ARRAY_TO_ACCEL(
                kBatchNormHw, "host_activations", activations, inputs_size);
        MAP_ARRAY_TO_ACCEL(
                kBatchNormHw, "host_weights", curr_layer_weights, weights_size);
        MAP_ARRAY_TO_ACCEL(kBatchNormHw, "host_result", result, outputs_size);
        INVOKE_KERNEL_PROF(kBatchNormHw, lnum, batch_norm_layer_hw, activations,
                           curr_layer_weights, result, g_umem, g_spad0, g_spad1,
                           &layers[lnum]);
    } else {
        begin_profiling(__func__, lnum);
        // By default, use the reference implementation.
        // TODO: Replace this with an MKL implementation after we've made one
        // that can take advantage of precomputed 1/sqrt(var).
        batch_norm_fxp(activations,
                       curr_layer_weights,
                       &layers[lnum],
                       NUM_TEST_CASES,
                       result);
        end_profiling();
    }
    return result;
}

result_buf smiv_activation_function(float* activations,
                                    layer_t* layer,
                                    float* results,
                                    device_t* device) {
#ifdef __cplusplus
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
    activation_fun(activations, NUM_TEST_CASES, output_size, layer->activation,
                   sigmoid_table);
    end_profiling();
    return activations;
#endif
}

result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     device_t* device,
                     sampling_param_t* sampling_param) {
    begin_profiling("run_layer", layer_num);

    begin_profiling("run_layer_skip_activation_func", layer_num);
    result_buf result_loc = run_layer_skip_activation_func(activations,
                                                           weights,
                                                           layers,
                                                           layer_num,
                                                           result,
                                                           device,
                                                           sampling_param);
    end_profiling();

    activation_type act_func = layers[layer_num].activation;
    bool do_activation = act_func != NO_ACTIVATION;
    bool do_hw_activation =
            device->use_hw_activation_func &&
            is_supported_activation_func(layers[layer_num].type, act_func);
    if (do_activation && !do_hw_activation) {
        if (result_loc == activations) {
            result_loc = smiv_activation_function(
                    activations, &layers[layer_num], result, device);
        } else {
            result_loc = smiv_activation_function(
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

// Set the dmaLoad/dmaStore required flags for each layer.
//
// Since SMIV can share scratchpads between the conv/fc blocks, we only need
// DMA if we need to send data back to the CPU.
void set_dma_requirements(network_t* network, device_t* device) {
    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        layer_t* curr_layer = &network->layers[layer_num];

        // The input layer is easy.
        if (layer_num == 0) {
            curr_layer->input_req = IO_NONE;
            curr_layer->output_req = IO_DMA;
            continue;
        }

        layer_t* prev_layer = &network->layers[layer_num - 1];
        layer_t* next_layer = &network->layers[layer_num + 1];
#if DEBUG_LEVEL > 0
        // When debugging, if we don't DMA the results back, we won't be able
        // to see what's happening.
        curr_layer->output_req = IO_DMA;
#else
        // First, determine if we need to dma store the output.
        if (layer_num == network->depth - 1 ||
            // All these activation functions are unsupported.
            curr_layer->activation == LRELU ||
            curr_layer->activation == ELU ||
            curr_layer->activation == SELU ||
            curr_layer->activation == TANH ||
            curr_layer->activation == SIGMOID ||
            curr_layer->activation == SOFTMAX ||
            // If we disabled HW activation functions but an activation
            // function is necessary, we need to DMA.
            (!device->use_hw_activation_func &&
             curr_layer->activation != NO_ACTIVATION) ||
            curr_layer->type == POOLING ||
            // For now, conv layers also do not support local caching.
            curr_layer->type == CONV_STANDARD ||
            curr_layer->type == CONV_DEPTHWISE ||
            curr_layer->type == CONV_POINTWISE ||
            curr_layer->type == BATCH_NORM ||
            next_layer->type == BATCH_NORM ||
            next_layer->type == POOLING ||
            // If the FC block needs work division, we can't locally cache.
            (curr_layer->type == FC && next_layer->type == FC &&
             inner_product_needs_work_division(&network->layers[layer_num]))) {
            curr_layer->output_req = IO_DMA;
        } else {
            curr_layer->output_req = IO_NONE;
        }
        if (curr_layer->input_preprocessing == FLATTEN)
            prev_layer->output_req = IO_DMA;
#endif
        // Whether we need to load the input on this layer is just whether we
        // had to store the outputs in the previous layer.
        curr_layer->input_req = prev_layer->output_req;
    }

    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        printf("Layer %d: dmaLoad = %d, dmaStore = %d\n", layer_num,
               network->layers[layer_num].input_req,
               network->layers[layer_num].output_req);
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

    g_umem = (float*)malloc_aligned(UMEM_SIZE);
    g_spad0 = (float*)malloc_aligned(SPAD_SIZE);
    g_spad1 = (float*)malloc_aligned(SPAD_SIZE);

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

    set_dma_requirements(&network, device);

    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_activations", activations.d,
                       activations.size);

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

    if (result_loc == result.d)
        dmaStore(result.d, result.d, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(activations.d, activations.d,
                 NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(network.layers, network.layers, network.depth * sizeof(layer_t));

    free(g_umem);
    free(g_spad0);
    free(g_spad1);
}

#endif
