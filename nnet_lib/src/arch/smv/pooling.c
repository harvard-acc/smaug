#include <assert.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
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

static void smv_pooling_layer_hw_impl(packed_fp16* host_activations,
                                      packed_fp16* host_results,
                                      float* local_activations,
                                      float* local_results,
                                      layer_t curr_layer,
                                      smv_pooling_options* options) {
    size_t partial_input_size =
            curr_layer.inputs.rows * curr_layer.inputs.cols *
            (curr_layer.inputs.height + curr_layer.inputs.align_pad);
    if (curr_layer.input_req == IO_DMA) {
        setReadyBits(local_activations, partial_input_size * sizeof(float), 0);
        dma_load_and_unpack_fp16(
                local_activations, host_activations, partial_input_size, 0, 0);
    } else if (curr_layer.input_req == IO_ACP) {
        acp_load_and_unpack_fp16(
                local_activations, host_activations, partial_input_size, 0, 0);
    }

    // TODO: This uses the existing SMIV pooling implementation, which only has
    // an 8-way SIMD datapath.
    if (curr_layer.pool == MAX)
        maxpooling_nhwc_smiv(local_activations, curr_layer, local_results);
    else
        avgpooling_nhwc_smiv(local_activations, curr_layer, local_results);

    size_t partial_output_size =
            curr_layer.outputs.rows * curr_layer.outputs.cols *
            (curr_layer.outputs.height + curr_layer.outputs.align_pad);
    if (curr_layer.output_req == IO_DMA) {
        dma_pack_and_store_fp16(
                host_results, local_results, partial_output_size, 0, 0);
    } else if (curr_layer.output_req == IO_ACP) {
        acp_pack_and_store_fp16(
                host_results, local_results, partial_output_size, 0, 0);
    }
}

static void smv_pooling_layer_hw(packed_fp16* dma_activations,
                                 packed_fp16* dma_results,
                                 packed_fp16* cache_activations,
                                 packed_fp16* cache_results,
                                 packed_fp16* acp_activations,
                                 packed_fp16* acp_results,
                                 float* umem,
                                 float* spad0,
                                 float* spad1,
                                 layer_t curr_layer,
                                 smv_pooling_options* options) {
    // We don't currently support using a local cache for inner products.  If
    // the IO requirement is IO_CACHE, it will be treated as IO_ACP.
    bool use_acp_results = (curr_layer.output_req == IO_ACP ||
                            curr_layer.output_req == IO_CACHE);
    bool use_acp_inputs = (curr_layer.input_req == IO_ACP ||
                           curr_layer.input_req == IO_CACHE);
    if (use_acp_results) {
        curr_layer.output_req = IO_ACP;
        if (use_acp_inputs) {
            curr_layer.input_req = IO_ACP;
            smv_pooling_layer_hw_impl(acp_activations, acp_results, spad0,
                                      spad1, curr_layer, options);
        } else {
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            smv_pooling_layer_hw_impl(dma_activations, acp_results, spad0,
                                      spad1, curr_layer, options);
        }
    } else {
        smv_pooling_layer_hw_impl(dma_activations, dma_results, spad0, spad1,
                                  curr_layer, options);
    }
}

void smv_pooling_layer_impl(data_list* inputs,
                            layer_t* curr_layer,
                            smv_global* g_smv,
                            data_list* results,
                            device_t* device) {
    require_data_type(inputs, 0, UncompressedHalfPrecision);
    pool_cfg_t pool_cfgs = smiv_pooling_divide_work(curr_layer);

    data_list* nhwc_inputs = init_data_list(1);
    begin_profiling("convert_nchw_to_blocked_nhwc", curr_layer->num);
    convert_nchw_to_blocked_nhwc(inputs, 0, NUM_TEST_CASES, VECTOR_SIZE,
                                 curr_layer->inputs, DATA_ALIGNMENT,
                                 nhwc_inputs);
    end_profiling();

    // Prepare a temporary buffer for the NHWC-formatted outputs.
    data_list* nhwc_outputs = init_data_list(1);
    nhwc_outputs->type[0] = inputs->type[0];
    nhwc_outputs->data[0].dense_hp = init_fp16array(
            compute_blocked_nhwc_size(
                    &curr_layer->outputs, VECTOR_SIZE, DATA_ALIGNMENT),
            true);

    for (int img = 0; img < NUM_TEST_CASES; img++) {
        packed_fp16* current_inputs = nhwc_inputs->data[0].dense_hp->d;
        packed_fp16* current_results = nhwc_outputs->data[0].dense_hp->d;
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
                        current_inputs, partial_input_size * sizeof(float16));
            }
            if (partial_layer.output_req == IO_DMA) {
                flush_cache_range(
                        current_results, partial_output_size * sizeof(float16));
            }
            end_profiling();

            MAP_ARRAY_TO_ACCEL(
                    g_smv->kPoolingHw,
                    get_host_inputs_var_name(partial_layer.input_req),
                    current_inputs,
                    partial_input_size * sizeof(float16));
            MAP_ARRAY_TO_ACCEL(
                    g_smv->kPoolingHw,
                    get_host_results_var_name(partial_layer.output_req),
                    current_results,
                    partial_output_size * sizeof(float16));

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

            current_inputs += partial_input_size / 2;
            current_results += partial_output_size / 2;
        }
    }

    dims_t output_dims =
            nchw_to_nhwc_dims(&curr_layer->outputs, DATA_ALIGNMENT);
    begin_profiling("convert_blocked_nhwc_to_nhwc", curr_layer->num);
    convert_blocked_nhwc_to_nchw(nhwc_outputs, 0, NUM_TEST_CASES, VECTOR_SIZE,
                                 output_dims, DATA_ALIGNMENT, results);
    end_profiling();
    free_data_list(nhwc_inputs);
    free_data_list(nhwc_outputs);
    free_smiv_work_cfg(&pool_cfgs);
}
