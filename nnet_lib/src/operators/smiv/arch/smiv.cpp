#include <assert.h>
#include <math.h>
#include <string.h>

#include "gem5/m5ops.h"

#include "nnet_fwd.h"
#include "core/ref/activation_functions.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
#include "core/smiv/smiv.h"
#include "core/smiv/params.h"
#include "utility/data_layout_conversion.h"
#include "utility/profiling.h"
#include "utility/utility.h"
#include "arch/common.h"
#include "arch/interface.h"
#include "arch/smiv/common.h"

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

smiv_global g_smiv;

void init_smiv_global(device_t* device) {
    // Use the same accelerator id for all hardware blocks. This means we will
    // simulate only ONE datapath instead of multiple, which means that the two
    // blocks can share the scratchpads (without any infrastructure
    // changes). The key is that we still trace the functions at the _hw level,
    // so that Aladdin will exit after simulating each block, and we can return
    // control to the CPU at the right places.  In contrast, if we used two
    // different ids, we would have two different datapaths that could not share
    // data directly.
    g_smiv.kConvolutionHw = 0x0003;
    g_smiv.kInnerProductHw = 0x0003;
    g_smiv.kReductionHw = 0x0003;
    g_smiv.kBatchNormHw = 0x0003;
    g_smiv.kPoolingHw = 0x0003;
    if (device->umem_size != 0) {
        g_smiv.kUmemSize = device->umem_size;
    } else {
        g_smiv.kUmemSize = SMIV_DEFAULT_UMEM_SIZE;
    }
    if (device->spad_size != 0) {
        g_smiv.kSpadSize = device->spad_size;
    } else {
        g_smiv.kSpadSize = SMIV_DEFAULT_SPAD_SIZE;
    }
    if (device->l2_size != 0) {
        g_smiv.kL2Size = device->l2_size;
    } else {
        g_smiv.kL2Size = SMIV_DEFAULT_L2_SIZE;
    }
    printf("Size of UMEM: %lu bytes\n", g_smiv.kUmemSize);
    printf("Size of Scratchpad: %lu bytes\n", g_smiv.kSpadSize);
    printf("Size of L2 cache: %lu bytes\n", g_smiv.kL2Size);

    g_smiv.umem = (float*)malloc_aligned(g_smiv.kUmemSize);
    g_smiv.spad0 = (float*)malloc_aligned(g_smiv.kSpadSize);
    g_smiv.spad1 = (float*)malloc_aligned(g_smiv.kSpadSize);
}

void free_smiv_global() {
    free(g_smiv.umem);
    free(g_smiv.spad0);
    free(g_smiv.spad1);
}

result_buf flatten_input(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results) {
    begin_profiling(__func__, lnum);
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    results = im2row(activations, layers, lnum, results);
    end_profiling();
    return results;
}

result_buf smiv_activation_function(data_list* activations,
                                    layer_t* layer,
                                    data_list* results,
                                    device_t* device) {
#ifdef SMIV_USE_MKL_ACTIVATION_FUNCTION_IMPL
    // MKL's implementation requires a separate output buffer.
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
#endif

    smiv_activation_function_impl(activations->data[0].dense->d, layer,
                                  results->data[0].dense->d, device);

#ifdef SMIV_USE_MKL_ACTIVATION_FUNCTION_IMPL
    return results;
#else
    // Our own implementation is in-place.
    return activations;
#endif
}


result_buf inner_product_layer(data_list* host_activations,
                               data_list* host_weights,
                               layer_t* layers,
                               int lnum,
                               data_list* host_results,
                               device_t* device,
                               sampling_param_t* sampling_param) {
    host_results = create_new_data_list_if_necessary(
            host_results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    smiv_inner_product_layer_impl(
            host_activations->data[0].dense->d, host_weights->data[0].dense->d,
            layers, lnum, host_results->data[0].dense->d, &g_smiv, device);
    return host_results;
}

result_buf standard_convolution_layer(data_list* activations,
                                      data_list* weights,
                                      layer_t* layers,
                                      int lnum,
                                      data_list* results,
                                      device_t* device,
                                      sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.weights.rows > VECTOR_SIZE ||
        curr_layer.weights.cols > VECTOR_SIZE) {
        fprintf(stderr,
                "[ERROR]: In layer %d: SMIV does not support "
                "convolutional weights with rows or cols > %d!\n",
                curr_layer.num, VECTOR_SIZE);
    }
    assert(curr_layer.weights.rows <= VECTOR_SIZE &&
           curr_layer.weights.cols <= VECTOR_SIZE);
    io_req_t input_req = layers[lnum].input_req;
    const char* weights_var_name =
            input_req == IO_DMA ? "host_weights" : input_req == IO_ACP
                                                           ? "acp_weights"
                                                           : "cache_weights";
    int weights_size = WEIGHT_BYTES(layers, lnum);
    MAP_ARRAY_TO_ACCEL(g_smiv.kConvolutionHw, weights_var_name,
                       weights->data[0].dense->d, weights_size);
    if (has_padding(&curr_layer.pad)) {
        results = create_new_data_list_if_necessary(
                results ,
                NUM_TEST_CASES * get_dims_size(&layers[lnum].inputs),
                Uncompressed);
        copy_zeropad(activations->data[0].dense->d, layers, lnum,
                     results->data[0].dense->d);
        PRINT_MSG("After zeropadding:\n");
        PRINT_DEBUG4D(results->data[0].dense->d,
                      curr_layer.inputs.rows,
                      curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                      curr_layer.inputs.height);
        PRINT_DEBUG4D_V(weights->data[0].dense->d, curr_layer.weights.rows,
                        curr_layer.weights.cols + curr_layer.weights.align_pad,
                        curr_layer.weights.height);
        SWAP_PTRS(results, activations);
    }
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    smiv_standard_convolution_layer_impl(
            activations->data[0].dense->d,
            weights->data[0].dense->d,
            layers,
            lnum,
            results->data[0].dense->d,
            &g_smiv,
            device,
            sampling_param);
    return results;
}

result_buf depthwise_convolution_layer(data_list* activations,
                                       data_list* weights,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    io_req_t input_req = layers[lnum].input_req;
    const char* weights_var_name =
            input_req == IO_DMA ? "host_weights" : input_req == IO_ACP
                                                           ? "acp_weights"
                                                           : "cache_weights";
    int weights_size = WEIGHT_BYTES(layers, lnum);
    MAP_ARRAY_TO_ACCEL(g_smiv.kConvolutionHw, weights_var_name,
                       weights->data[0].dense->d, weights_size);
    layer_t curr_layer = layers[lnum];
    if (has_padding(&curr_layer.pad)) {
        results = create_new_data_list_if_necessary(
                results,
                NUM_TEST_CASES * get_dims_size(&layers[lnum].inputs),
                Uncompressed);
        copy_zeropad(activations->data[0].dense->d, layers, lnum,
                     results->data[0].dense->d);
        PRINT_MSG("After zeropadding:\n");
        PRINT_DEBUG4D(results->data[0].dense->d,
                      curr_layer.inputs.rows,
                      curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                      curr_layer.inputs.height);
        PRINT_DEBUG4D_V(weights->data[0].dense->d, curr_layer.weights.rows,
                        curr_layer.weights.cols + curr_layer.weights.align_pad,
                        curr_layer.weights.height);
        SWAP_PTRS(activations, results);
    }
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    smiv_depthwise_convolution_layer_impl(
            activations->data[0].dense->d, weights->data[0].dense->d, layers,
            lnum, results->data[0].dense->d, &g_smiv, device);

    return results;
}

// SMIV currently uses the FC block to implement a GEMM-based 1x1 convolution
// because the current model of SMIV in Aladdin doesn't support 1x1 conv on the
// CONV block.  Eventually we'll want to use the CNN block since the CNN block
// outputs results in NCHW format (where as the FC block outputs data in NHWC
// format).
result_buf pointwise_convolution_layer(data_list* activations,
                                       data_list* weights,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    // Allocate memory to store the transformed input.
    float* nhwc_inputs = NULL;
    dims_t nhwc = convert_nchw_to_nhwc_fp32(activations->data[0].dense->d,
                                            NUM_TEST_CASES, layers[lnum].inputs,
                                            DATA_ALIGNMENT, &nhwc_inputs);

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

    // Allocate new memory to store the results of the FC. The
    // activations/results buffers are not necessarily big enough to store this
    // (due to data alignment).
    float* nhwc_outputs = (float*)malloc_aligned(
            get_dims_size(&layers[lnum].outputs) * sizeof(float));

    // Finally, invoke the FC hardware.
    smiv_inner_product_layer_impl(nhwc_inputs, weights->data[0].dense->d,
                                  layers, lnum, nhwc_outputs, &g_smiv, device);

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
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    convert_nhwc_to_nchw_fp32(nhwc_outputs, NUM_TEST_CASES, output_dims,
                              DATA_ALIGNMENT, &results->data[0].dense->d);

    // Restore the original layer descriptor.
    layers[lnum] = old_layer;

    free(nhwc_inputs);
    free(nhwc_outputs);
    return results;
}

// Software implementation. SMIV doesn't accelerate pooling.
result_buf pooling_layer(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results,
                         device_t* device,
                         sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    if (device->use_hw_pooling) {
        smiv_pooling_layer_impl(activations, &layers[lnum], &g_smiv, results);
    } else {
#ifdef __cplusplus
        if (curr_layer.pool == MAX) {
            nnet_mkl::max_pooling_3d(
                    act_buf, &layers[lnum], out_buf, device);
        } else if (curr_layer.pool == AVG) {
            nnet_mkl::avg_pooling_3d(
                    act_buf, &layers[lnum], out_buf, device);
        } else {
            assert(false && "Unsupported pooling layer type!");
        }
        nnet_mkl::MklSession* session = nnet_mkl::get_session(device);
        session->run_and_clear();
#else
        // This code should only get run by the tracer.
        if (curr_layer.pool == MAX) {
            max_pooling(act_buf, out_buf, layers[lnum]);
        } else if (curr_layer.pool == AVG) {
            avg_pooling(act_buf, out_buf, layers[lnum]);
        } else {
            assert(false && "Unsupported pooling layer type!");
        }
#endif
    }
    return results;
}

result_buf batch_norm_layer(data_list* activations,
                            data_list* weights,
                            layer_t* layers,
                            int lnum,
                            data_list* results,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    smiv_batch_norm_layer_impl(activations->data[0].dense->d,
                               weights->data[0].dense->d, layers, lnum,
                               results->data[0].dense->d, &g_smiv, device);
    return results;
}


result_buf run_layer(data_list* activations,
                     data_list* weights,
                     layer_t* layers,
                     int layer_num,
                     data_list* results,
                     device_t* device,
                     sampling_param_t* sampling_param) {
    begin_profiling("run_layer", layer_num);

    begin_profiling("layer_dispatcher", layer_num);
    result_buf result_loc = layer_dispatcher(activations,
                                             weights,
                                             layers,
                                             layer_num,
                                             results,
                                             device,
                                             sampling_param);
    end_profiling();

    activation_type act_func = layers[layer_num].activation;
    bool do_activation = act_func != NO_ACTIVATION;
    bool do_hw_activation =
            device->use_hw_activation_func &&
            smiv_is_supported_activation_func(layers[layer_num].type, act_func);
    if (do_activation && !do_hw_activation) {
        if (result_loc == results) {
            SWAP_PTRS(activations, results);
        }
        result_loc = smiv_activation_function(
                activations, &layers[layer_num], results, device);
        PRINT_MSG("\nactivation function\n");
        PRINT_DEBUG4D(result_loc->data[0].dense->d,
                      layers[layer_num].outputs.rows,
                      layers[layer_num].outputs.cols +
                              layers[layer_num].outputs.align_pad,
                      layers[layer_num].outputs.height);
    }
    end_profiling();
    dump_profiling_log();
    return result_loc;
}

// Set the IO required flags for each layer.
//
// Since SMIV can share scratchpads between the conv/fc blocks, we only need
// IO if we need to send data back to the CPU.
void set_io_requirements(network_t* network,
                         device_t* device,
                         smiv_global* g_smiv) {
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

        // We only support DMA for hardware batch norm and pooling layers.
        if (curr_layer->type == BATCH_NORM || curr_layer->type == POOLING) {
            curr_layer->input_req = IO_DMA;
            curr_layer->output_req = IO_DMA;
            continue;
        }

        // First, determine if we need to dma store the output.
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
            next_layer->type == BATCH_NORM || next_layer->type == POOLING ||
            // If the FC block needs work division, we can't locally cache.
            (curr_layer->type == FC && next_layer->type == FC &&
             smiv_inner_product_needs_work_division(
                     &network->layers[layer_num], g_smiv))) {
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
        // flattening, it means the previous layer needs to send resutls
        // back to the CPU.
        if (curr_layer->input_preprocessing == FLATTEN) {
            // Since batch norm and pooling blocks only support DMA, so except
            // them here.
            if (!(prev_layer->type == BATCH_NORM) &&
                !(prev_layer->type == POOLING))
                prev_layer->output_req = device->cpu_default_offload;
        }
        // If the previous layer doesn't need to send back results (e.g.,
        // FC->FC caching), the current layer needs no IO for inputs.
        if (prev_layer->output_req == IO_NONE) {
            curr_layer->input_req = IO_NONE;
        } else {
            curr_layer->input_req = device->cpu_default_offload;
        }
#endif
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
// ping-ponged between two buffers, activations and results.
void nnet_fwd(data_list* activations,
              data_list* weights,
              data_list* results,
              network_t* network,
              device_t* device,
              sampling_param_t* sampling_param) {
    init_smiv_global(device);

    M5_SWITCH_CPU();

#ifdef __cplusplus
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device->session = (void*)session;
#endif
    set_io_requirements(network, device, &g_smiv);

    //******************//
    //   PRIMARY LOOP   //
    //******************//

    // We need to ensure that we update the original data_list objects, but
    // internally a lot of pointer-swapping is done to reduce the number of
    // memory allocations, so to separate these two worlds, create internal
    // copies.
    data_list* activations_internal = activations;
    data_list* results_internal = results;

    // Alternate between reading from/writing to activations and results so we
    // can avoid copying matrices. The initial activations is obviously in
    // "activations", so that's where we start.
    result_buf result_loc = activations;
    nnet_fwd_outer:
    for (int l = 0; l < network->depth; l++) {
        if (result_loc == results_internal) {
            SWAP_PTRS(results_internal, activations_internal);
        }
        result_loc = run_layer(
                activations_internal, network->layers[l].host_weights,
                network->layers, l, results_internal, device, sampling_param);
    }

    results = copy_data_list(results, result_loc);
    network->layers[network->depth - 1].result_in_temp = true;

    free_smiv_global();
}

#endif
