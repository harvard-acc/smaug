#include "arch/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/smiv/smiv.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _smv_batch_norm_options {
    bool input_in_spad0;
    bool use_pipelined_dma;
} smv_batch_norm_options;

static void smv_batch_norm_layer_hw_impl(float* host_activations,
                                         packed_fp16* host_weights,
                                         float* host_results,
                                         float* local_activations,
                                         float* local_weights,
                                         float* local_results,
                                         layer_t* curr_layer,
                                         smv_batch_norm_options* options) {
    // DMA in the weights (to UMEM).
    int weights_size = get_num_weights_layer(curr_layer, 0) * sizeof(float);
    setReadyBits(local_weights, weights_size, 0);
    dma_load_and_unpack_fp16(local_weights, host_weights, weights_size, 0, 0);

    // Load in the inputs.
    int input_size = get_input_activations_size(curr_layer) * sizeof(float);
    if (curr_layer->input_req == IO_DMA) {
        setReadyBits(local_activations, input_size, 0);
        dma_load_wrapper(local_activations, host_activations, input_size,
                         options->use_pipelined_dma);
    } else if (curr_layer->input_req == IO_ACP) {
        coherentLoad64(local_activations, host_activations, input_size, 0, 0);
    }

    // The main kernel
#ifdef ENABLE_SIMD_IMPL
    batch_norm_simd_fxp(local_activations, local_weights, curr_layer,
                        NUM_TEST_CASES, local_results);
#else
    activation_type activation = curr_layer->activation;
    batch_norm_fxp(local_activations, local_weights, curr_layer, NUM_TEST_CASES,
                   local_results);
    activation_fun(local_results, NUM_TEST_CASES, input_size, activation);
#endif

    // DMA out the result (from SPAD1)
    int output_size = get_output_activations_size(curr_layer) * sizeof(float);
    if (curr_layer->output_req == IO_DMA) {
        dma_store_wrapper(host_results, local_results, output_size,
                          options->use_pipelined_dma);
    } else if (curr_layer->output_req == IO_ACP) {
        coherentStore64(host_results, local_results, output_size, 0, 0);
    }
}

static void smv_batch_norm_layer_hw(float* dma_activations,
                                    packed_fp16* dma_weights,
                                    float* dma_results,
                                    float* cache_activations,
                                    packed_fp16* cache_weights,
                                    float* cache_results,
                                    float* acp_activations,
                                    packed_fp16* acp_weights,
                                    float* acp_results,
                                    float* umem,
                                    float* spad0,
                                    float* spad1,
                                    layer_t* curr_layer,
                                    smv_batch_norm_options* options) {
    // We can use ACP or caches for outputs only, or for outputs AND inputs,
    // but not inputs only. We also do not currently support mixing ACP and
    // cache for the inputs/outputs.
    if (curr_layer->output_req == IO_ACP) {
        if (curr_layer->input_req == IO_ACP) {
            smv_batch_norm_layer_hw_impl(acp_activations, dma_weights,
                                         acp_results, umem, spad0, spad1,
                                         curr_layer, options);
        } else {
            // If the input mechanism is Cache, then it is ignored, and we
            // fallback to DMA.
            if (curr_layer->input_req != IO_NONE)
                curr_layer->input_req = IO_DMA;
            smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                         dma_results, umem, spad0, spad1,
                                         curr_layer, options);
        }
    } else if (curr_layer->output_req == IO_CACHE) {
        if (curr_layer->input_req == IO_CACHE) {
            smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                         dma_results, cache_activations, spad0,
                                         cache_results, curr_layer, options);
        } else {
            // If the input mechanism is ACP, then it is ignored, and we
            // fallback to DMA.
            if (curr_layer->input_req != IO_NONE)
                curr_layer->input_req = IO_DMA;
            smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                         dma_results, umem, spad0,
                                         cache_results, curr_layer, options);
        }
    } else {
        if (curr_layer->input_req != IO_NONE)
            curr_layer->input_req = IO_DMA;
        smv_batch_norm_layer_hw_impl(dma_activations, dma_weights, dma_results,
                                     umem, spad0, spad1, curr_layer, options);
    }

}

void smv_batch_norm_layer_impl(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result,
                               smv_global* g_smv,
                               device_t* device) {
    layer_t curr_layer = layers[lnum];

    if (device->use_hw_batch_norm) {
        assert(curr_layer.host_weights.type[0] == UncompressedHalfPrecision);
        uarray_t* bn_weights = curr_layer.host_weights.data[0].dense_hp;
        int weights_size = bn_weights->size * sizeof(short);
        if (weights_size > SMV_UMEM_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm weights are larger than the "
                            "UMEM - not currently supported!\n");
        }
        assert(weights_size <= SMV_UMEM_SIZE);
        int inputs_size = INPUT_BYTES(layers, lnum);
        int outputs_size = OUTPUT_BYTES(layers, lnum);
        assert(inputs_size == outputs_size);
        if (inputs_size > SMV_SPAD_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm inputs don't fit on the "
                            "scratchpad!\n");
        }
        assert(inputs_size <= SMV_SPAD_SIZE);
        if (!device->use_hw_activation_func)
            curr_layer.activation = NO_ACTIVATION;

        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        if (curr_layer.input_req == IO_DMA) {
            flush_cache_range(activations, inputs_size);
        }
        if (curr_layer.weights_req == IO_DMA) {
            flush_cache_range(bn_weights->d, weights_size);
        }
        end_profiling();

        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_inputs_var_name(curr_layer.input_req),
                           activations, inputs_size);
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_weights_var_name(curr_layer.weights_req),
                           bn_weights->d, weights_size);
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_results_var_name(curr_layer.output_req),
                           result, outputs_size);
        // TODO: For now, always put the input into spad0.
        smv_batch_norm_options options;
        options.input_in_spad0 = true;
        options.use_pipelined_dma = device->use_pipelined_dma;
        INVOKE_KERNEL_PROF(g_smv->kBatchNormHw, lnum, smv_batch_norm_layer_hw,
                           activations, bn_weights->d, result,  // DMA
                           activations, bn_weights->d, result,  // Cache
                           activations, bn_weights->d, result,  // ACP
                           g_smv->umem, g_smv->spad0, g_smv->spad1,
                           &curr_layer, &options);
    } else {
        begin_profiling(__func__, lnum);
        // The reference implementation is faster than MKL since we can
        // precompute some of the weights. We have an optimized MKL version,
        // but it just calls this same function, so there's no point going
        // through that overhead.
        float* bn_weights = weights + get_weights_loc_for_layer(layers, lnum);
        batch_norm_fxp(activations,
                       bn_weights,
                       &curr_layer,
                       NUM_TEST_CASES,
                       result);
        if (device->use_hw_activation_func) {
            int input_size = get_dims_size(&curr_layer.inputs);
            activation_fun(
                    result, NUM_TEST_CASES, input_size, curr_layer.activation);
        }
        end_profiling();
    }
}
