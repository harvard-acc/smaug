#include "arch/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _smv_batch_norm_options {
    bool input_in_spad0;
    bool use_pipelined_dma;
} smv_batch_norm_options;

static void smv_batch_norm_layer_hw_impl(packed_fp16* host_activations,
                                         packed_fp16* host_weights,
                                         packed_fp16* host_results,
                                         float* local_activations,
                                         float* local_weights,
                                         float* local_results,
                                         layer_t* curr_layer,
                                         smv_batch_norm_options* options) {
    // DMA in the weights (to UMEM).
    int weights_size = get_num_weights_layer(curr_layer, 0);
    setReadyBits(local_weights, weights_size, 0);
    dma_load_and_unpack_fp16(local_weights, host_weights, weights_size, 0, 0);

    // Load in the inputs.
    int input_size = get_input_activations_size(curr_layer);
    if (curr_layer->input_req == IO_DMA) {
        setReadyBits(local_activations, input_size, 0);
        dma_load_and_unpack_fp16(
                local_activations, host_activations, input_size, 0, 0);
    } else if (curr_layer->input_req == IO_ACP) {
        acp_load_and_unpack_fp16(
                local_activations, host_activations, input_size, 0, 0);
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
    int output_size = get_output_activations_size(curr_layer);
    if (curr_layer->output_req == IO_DMA) {
        dma_pack_and_store_fp16(host_results, local_results, output_size, 0, 0);
    } else if (curr_layer->output_req == IO_ACP) {
        acp_pack_and_store_fp16(host_results, local_results, output_size, 0, 0);
    }
}

static void smv_batch_norm_layer_hw(packed_fp16* dma_activations,
                                    packed_fp16* dma_weights,
                                    packed_fp16* dma_results,
                                    packed_fp16* cache_activations,
                                    packed_fp16* cache_weights,
                                    packed_fp16* cache_results,
                                    packed_fp16* acp_activations,
                                    packed_fp16* acp_weights,
                                    packed_fp16* acp_results,
                                    float* umem,
                                    float* spad0,
                                    float* spad1,
                                    layer_t* curr_layer,
                                    smv_batch_norm_options* options) {
    bool use_acp_results = (curr_layer->output_req == IO_ACP ||
                            curr_layer->output_req == IO_CACHE);
    bool use_acp_inputs = (curr_layer->input_req == IO_ACP ||
                           curr_layer->input_req == IO_CACHE);
    if (use_acp_results) {
        curr_layer->output_req = IO_ACP;
        if (use_acp_inputs) {
            curr_layer->input_req = IO_ACP;
            smv_batch_norm_layer_hw_impl(acp_activations, dma_weights,
                                         acp_results, umem, spad0, spad1,
                                         curr_layer, options);
        } else {
            if (curr_layer->input_req != IO_NONE)
                curr_layer->input_req = IO_DMA;
            smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                         acp_results, umem, spad0, spad1,
                                         curr_layer, options);
        }
    } else {
        if (curr_layer->input_req != IO_NONE)
            curr_layer->input_req = IO_DMA;
        smv_batch_norm_layer_hw_impl(dma_activations, dma_weights, dma_results,
                                     umem, spad0, spad1, curr_layer, options);
    }

}

void smv_batch_norm_layer_impl(data_list* activations,
                               layer_t* layers,
                               int lnum,
                               data_list* results,
                               smv_global* g_smv,
                               device_t* device) {
    layer_t curr_layer = layers[lnum];
    data_list* weights = curr_layer.host_weights;

    if (device->use_hw_batch_norm) {
        require_data_type(activations, 0, UncompressedHalfPrecision);
        require_data_type(weights, 0, UncompressedHalfPrecision);
        fp16array_t* bn_weights = weights->data[0].dense_hp;
        int weights_size = bn_weights->size * sizeof(short);
        if (weights_size > SMV_UMEM_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm weights are larger than the "
                            "UMEM - not currently supported!\n");
        }
        assert(weights_size <= SMV_UMEM_SIZE);
        int inputs_size = get_dims_size(&curr_layer.inputs) * NUM_TEST_CASES;
        int outputs_size = get_dims_size(&curr_layer.outputs) * NUM_TEST_CASES;
        assert(inputs_size == outputs_size);
        if (inputs_size * sizeof(float) > SMV_SPAD_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm inputs don't fit on the "
                            "scratchpad!\n");
        }
        assert(inputs_size * sizeof(float) <= SMV_SPAD_SIZE);
        if (!device->use_hw_activation_func)
            curr_layer.activation = NO_ACTIVATION;

        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        if (curr_layer.input_req == IO_DMA) {
            flush_cache_range(activations, inputs_size * sizeof(float16));
        }
        if (curr_layer.weights_req == IO_DMA) {
            flush_cache_range(bn_weights->d, weights_size * sizeof(float16));
        }
        end_profiling();

        packed_fp16* act_buf = activations->data[0].dense_hp->d;
        packed_fp16* wgt_buf = weights->data[0].dense_hp->d;
        packed_fp16* out_buf = results->data[0].dense_hp->d;
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_inputs_var_name(curr_layer.input_req),
                           act_buf,
                           inputs_size * sizeof(float16));
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_weights_var_name(curr_layer.weights_req),
                           wgt_buf, weights_size * sizeof(float16));
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_results_var_name(curr_layer.output_req),
                           out_buf,
                           outputs_size * sizeof(float16));
        // TODO: For now, always put the input into spad0.
        smv_batch_norm_options options;
        options.input_in_spad0 = true;
        options.use_pipelined_dma = device->use_pipelined_dma;
        INVOKE_KERNEL_PROF(g_smv->kBatchNormHw, lnum, smv_batch_norm_layer_hw,
                           act_buf, wgt_buf, out_buf,  // DMA
                           act_buf, wgt_buf, out_buf,  // Cache
                           act_buf, wgt_buf, out_buf,  // ACP
                           g_smv->umem, g_smv->spad0, g_smv->spad1,
                           &curr_layer, &options);
    } else {
        begin_profiling(__func__, lnum);
        // The reference implementation is faster than MKL since we can
        // precompute some of the weights. We have an optimized MKL version,
        // but it just calls this same function, so there's no point going
        // through that overhead.
        require_data_type(weights, 0, Uncompressed);
        float* bn_weights = weights->data[0].dense->d;
        farray_t* fp32_activations = NULL;
        farray_t* fp32_results = NULL;
        if (activations->type[0] == UncompressedHalfPrecision) {
            fp32_activations =
                    unpack_data_fp16x4(activations->data[0].dense_hp, NULL);
            fp32_results = init_farray(
                    NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
                    false);
        } else {
            fp32_activations = activations->data[0].dense;
            fp32_results = results->data[0].dense;
        }

        batch_norm_fxp(fp32_activations->d, bn_weights, &curr_layer,
                       NUM_TEST_CASES, fp32_results->d);
        if (device->use_hw_activation_func) {
            int input_size = get_dims_size(&curr_layer.inputs);
            activation_fun(fp32_results->d, NUM_TEST_CASES,
                           input_size, curr_layer.activation);
        }

        if (activations->type[0] == UncompressedHalfPrecision) {
            // Ugly - need to free the farray_t* container without freeing the
            // buffer it wraps.
            packed_fp16* results_buf = results->data[0].dense_hp->d;
            free(results->data[0].dense_hp);
            results->data[0].dense_hp =
                    pack_data_fp16(fp32_results, results_buf);
            free_farray(fp32_activations);
            free_farray(fp32_results);
        }
        end_profiling();
    }
}
