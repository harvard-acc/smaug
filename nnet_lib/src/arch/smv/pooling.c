#include <assert.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/smiv.h"
#include "core/smv/params.h"
#include "utility/data_layout_conversion.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _smv_pooling_options {
    bool use_pipelined_dma;
} smv_pooling_options;

static void smv_pooling_layer_hw_impl(float* host_activations,
                                      float* host_results,
                                      float* local_activations,
                                      float* local_results,
                                      layer_t curr_layer,
                                      smv_pooling_options* options) {
    size_t partial_input_size =
            curr_layer.inputs.rows * curr_layer.inputs.cols *
            (curr_layer.inputs.height + curr_layer.inputs.align_pad) *
            sizeof(float);
    if (curr_layer.input_req == IO_DMA) {
        setReadyBits(local_activations, partial_input_size, 0);
        dma_load_wrapper(local_activations,
                         host_activations,
                         partial_input_size,
                         options->use_pipelined_dma);
    } else if (curr_layer.input_req == IO_ACP) {
        coherentLoad64(
                local_activations, host_activations, partial_input_size, 0, 0);
    }

    // TODO: Use the existing SMIV pooling implementation, which only has an
    // 8-way SIMD datapath.
    if (curr_layer.pool == MAX)
        maxpooling_nhwc_smiv(local_activations, curr_layer, local_results);
    else
        avgpooling_nhwc_smiv(local_activations, curr_layer, local_results);

    size_t partial_output_size =
            curr_layer.outputs.rows * curr_layer.outputs.cols *
            (curr_layer.outputs.height + curr_layer.outputs.align_pad) *
            sizeof(float);
    if (curr_layer.output_req == IO_DMA) {
        dma_store_wrapper(host_results,
                          local_results,
                          partial_output_size,
                          options->use_pipelined_dma);
    } else if (curr_layer.output_req == IO_ACP) {
        coherentStore64(host_results, local_results, partial_output_size, 0, 0);
    }
}

static void smv_pooling_layer_hw(float* dma_activations,
                                 float* dma_results,
                                 float* cache_activations,
                                 float* cache_results,
                                 float* acp_activations,
                                 float* acp_results,
                                 float* umem,
                                 float* spad0,
                                 float* spad1,
                                 layer_t curr_layer,
                                 smv_pooling_options* options) {
    // We can use ACP or caches for outputs only, or for outputs AND inputs,
    // but not inputs only. We also do not currently support mixing ACP and
    // cache for the inputs/outputs.
    if (curr_layer.output_req == IO_ACP) {
        if (curr_layer.input_req == IO_ACP) {
            smv_pooling_layer_hw_impl(acp_activations, acp_results, spad0,
                                      spad1, curr_layer, options);
        } else {
            // If the input mechanism is Cache, then it is ignored, and we
            // fallback to DMA.
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            smv_pooling_layer_hw_impl(dma_activations, acp_results, spad0,
                                      spad1, curr_layer, options);
        }
    } else if (curr_layer.output_req == IO_CACHE) {
        if (curr_layer.input_req == IO_CACHE) {
            smv_pooling_layer_hw_impl(dma_activations, dma_results,
                                      cache_activations, cache_results,
                                      curr_layer, options);
        } else {
            // If the input mechanism is ACP, then it is ignored, and we
            // fallback to DMA.
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            smv_pooling_layer_hw_impl(dma_activations, dma_results,
                                      cache_activations, spad0, curr_layer,
                                      options);
        }
    } else {
        smv_pooling_layer_hw_impl(dma_activations, dma_results, spad0, spad1,
                                  curr_layer, options);
    }

}

void smv_pooling_layer_impl(float* inputs,
                             layer_t* curr_layer,
                             smv_global* g_smv,
                             float* results,
                             device_t* device) {
    pool_cfg_t pool_cfgs = smiv_pooling_divide_work(curr_layer);

    float* nhwc_inputs = NULL;
    begin_profiling("convert_nchw_to_blocked_nhwc", curr_layer->num);
    convert_nchw_to_blocked_nhwc(inputs,
                                 NUM_TEST_CASES,
                                 VECTOR_SIZE,
                                 curr_layer->inputs,
                                 DATA_ALIGNMENT,
                                 &nhwc_inputs);
    end_profiling();

    // Prepare a temporary buffer for the NHWC-formatted outputs.
    float* nhwc_outputs = (float*)malloc_aligned(
            compute_blocked_nhwc_size(
                    &curr_layer->outputs, VECTOR_SIZE, DATA_ALIGNMENT) *
            sizeof(float));

    for (int img = 0; img < NUM_TEST_CASES; img++) {
        float* current_inputs = nhwc_inputs;
        float* current_results = nhwc_outputs;
        for (unsigned iter = 0; iter < pool_cfgs.num_iterations; iter++) {
            PRINT_MSG("Iteration %d\n", iter);
            dims_t iter_cfg = pool_cfgs.iteration[iter];
            layer_t partial_layer = *curr_layer;
            partial_layer.inputs.height = iter_cfg.height;
            partial_layer.inputs.align_pad =
                    calc_padding(partial_layer.inputs.height, DATA_ALIGNMENT);
            partial_layer.outputs.height = iter_cfg.height;
            partial_layer.outputs.align_pad =
                    calc_padding(partial_layer.outputs.height, DATA_ALIGNMENT);
            size_t partial_input_size = get_dims_size(&partial_layer.inputs);
            size_t partial_output_size = get_dims_size(&partial_layer.outputs);

            // Flush cache lines for inputs.
            begin_ignored_profiling(partial_layer.num);
            if (partial_layer.input_req == IO_DMA) {
                flush_cache_range(
                        current_inputs, partial_input_size * sizeof(float));
            }
            if (partial_layer.output_req == IO_DMA) {
                flush_cache_range(
                        current_results, partial_output_size * sizeof(float));
            }
            end_profiling();

            MAP_ARRAY_TO_ACCEL(g_smv->kPoolingHw,
                               get_host_inputs_var_name(partial_layer.input_req),
                               current_inputs,
                               partial_input_size * sizeof(float));
            MAP_ARRAY_TO_ACCEL(g_smv->kPoolingHw,
                               get_host_results_var_name(partial_layer.output_req),
                               current_results,
                               partial_output_size * sizeof(float));

            smv_pooling_options options;
            options.use_pipelined_dma = device->use_pipelined_dma;
            INVOKE_KERNEL_PROF(g_smv->kPoolingHw,
                               partial_layer.num,
                               smv_pooling_layer_hw,
                               current_inputs,  // DMA
                               current_results,
                               current_inputs,  // Cache
                               current_results,
                               current_inputs,  // ACP
                               current_results,
                               g_smv->umem,
                               g_smv->spad0,
                               g_smv->spad1,
                               partial_layer,
                               &options);

            current_inputs += partial_input_size;
            current_results += partial_output_size;
        }
    }

    dims_t output_dims =
            nchw_to_nhwc_dims(&curr_layer->outputs, DATA_ALIGNMENT);
    begin_profiling("convert_blocked_nhwc_to_nhwc", curr_layer->num);
    convert_blocked_nhwc_to_nchw(nhwc_outputs,
                                 NUM_TEST_CASES,
                                 VECTOR_SIZE,
                                 output_dims,
                                 DATA_ALIGNMENT,
                                 &results);
    end_profiling();
    free(nhwc_inputs);
    free(nhwc_outputs);
}
