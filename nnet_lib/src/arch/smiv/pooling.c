#include <assert.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/smiv.h"
#include "core/smiv/params.h"
#include "utility/data_layout_conversion.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

static void smiv_pooling_layer_hw(float* host_activations,
                                  float* host_results,
                                  float* umem,
                                  float* spad0,
                                  float* spad1,
                                  layer_t partial_layer,
                                  int iteration_offset) {
    if (partial_layer.input_req == IO_DMA) {
        size_t partial_input_size =
                partial_layer.inputs.rows * partial_layer.inputs.cols *
                (partial_layer.inputs.height + partial_layer.inputs.align_pad);
        setReadyBits(spad0, partial_input_size * sizeof(float), 0);
        dmaLoad(spad0, host_activations, partial_input_size * sizeof(float));
    }

    if (partial_layer.pool == MAX)
        maxpooling_nhwc_smiv(spad0, partial_layer, 0, spad1);
    else
        avgpooling_nhwc_smiv(spad0, partial_layer, 0, spad1);

    if (partial_layer.output_req == IO_DMA) {
        size_t partial_output_size = partial_layer.outputs.rows *
                                     partial_layer.outputs.cols *
                                     (partial_layer.outputs.height +
                                      partial_layer.outputs.align_pad);
        dmaStore(host_results, spad1, partial_output_size * sizeof(float));
    }
}

// Pooling work division.
pool_cfg_t smiv_pooling_divide_work(layer_t* curr_layer, smiv_global* g_smiv) {
    pool_cfg_t pool_cfgs;
    dims_t input_nhwc_dims =
            nchw_to_nhwc_dims(&curr_layer->inputs, DATA_ALIGNMENT);
    unsigned total_input_bytes = get_dims_size(&input_nhwc_dims) * sizeof(float);
    if (total_input_bytes > g_smiv->kUmemSize) {
        fprintf(stderr,
                "A single input image exceeds the capacity of the UMEM, which "
                "is unsupported!\n");
        assert(false);
    }
    if (total_input_bytes <= g_smiv->kSpadSize) {
        PRINT_MSG_V("Entire input problem fits into the local memory.\n");
        init_smiv_work_cfg(&pool_cfgs, 1);
        pool_cfgs.iteration[0].rows = curr_layer->inputs.rows;
        pool_cfgs.iteration[0].cols = curr_layer->inputs.cols;
        pool_cfgs.iteration[0].height = curr_layer->inputs.height;
        pool_cfgs.iteration[0].align_pad =
                calc_padding(pool_cfgs.iteration[0].cols, DATA_ALIGNMENT);
        return pool_cfgs;
    }

    // Divide the problem up per groups of VECTOR_SIZE input channels.

    const unsigned input_block_size =
            VECTOR_SIZE * curr_layer->inputs.rows *
            (curr_layer->inputs.cols + curr_layer->inputs.align_pad) *
            sizeof(float);

    if (input_block_size > g_smiv->kSpadSize) {
        fprintf(stderr, "Tiled input handling is not yet supported!\n");
        assert(false);
    }

    const int num_blocks_per_iter = g_smiv->kSpadSize / input_block_size;
    const int num_channels_per_iter = num_blocks_per_iter * VECTOR_SIZE;
    const int total_channels = curr_layer->inputs.height;
    const int num_iterations =
            ceil(((float)total_channels) / num_channels_per_iter);

    init_smiv_work_cfg(&pool_cfgs, num_iterations);
    int remaining_channels = total_channels;
    for (int i = 0; i < num_iterations; i++) {
        pool_cfgs.iteration[i].rows = curr_layer->inputs.rows;
        pool_cfgs.iteration[i].cols = curr_layer->inputs.cols;
        pool_cfgs.iteration[i].height =
                min2(remaining_channels, num_channels_per_iter);
        pool_cfgs.iteration[i].align_pad =
                calc_padding(pool_cfgs.iteration[i].cols, DATA_ALIGNMENT);
        remaining_channels -= num_channels_per_iter;
    }
    return pool_cfgs;
}

void smiv_pooling_layer_impl(data_list* inputs,
                             layer_t* curr_layer,
                             smiv_global* g_smiv,
                             data_list* results) {
    pool_cfg_t pool_cfgs = smiv_pooling_divide_work(curr_layer, g_smiv);

    data_list* nhwc_inputs = init_data_list(1);
    begin_profiling("convert_nchw_to_blocked_nhwc", curr_layer->num);
    convert_nchw_to_blocked_nhwc(inputs, 0, NUM_TEST_CASES, VECTOR_SIZE,
                                 curr_layer->inputs, DATA_ALIGNMENT,
                                 nhwc_inputs);
    end_profiling();

    // Prepare a temporary buffer for the NHWC-formatted outputs.
    data_list* nhwc_outputs = init_data_list(1);
    nhwc_outputs->type[0] = inputs->type[0];
    nhwc_outputs->data[0].dense =
            init_farray(compute_blocked_nhwc_size(&curr_layer->outputs,
                                                  VECTOR_SIZE, DATA_ALIGNMENT),
                        true);

    for (int img = 0; img < NUM_TEST_CASES; img++) {
        float* current_inputs = nhwc_inputs->data[0].dense->d;
        float* current_results = nhwc_outputs->data[0].dense->d;
        int iteration_offset = 0;
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
            partial_layer.input_req = IO_DMA;
            partial_layer.output_req = IO_DMA;
            size_t partial_input_size = get_dims_size(&partial_layer.inputs);
            size_t partial_output_size = get_dims_size(&partial_layer.outputs);

            // Flush cache lines for inputs.
            begin_ignored_profiling(curr_layer->num);
            flush_cache_range(
                    current_inputs, partial_input_size * sizeof(float));
            end_profiling();

            MAP_ARRAY_TO_ACCEL(g_smiv->kPoolingHw,
                               "host_activations",
                               current_inputs,
                               partial_input_size * sizeof(float));
            MAP_ARRAY_TO_ACCEL(g_smiv->kPoolingHw,
                               "host_results",
                               current_results,
                               partial_output_size * sizeof(float));

            INVOKE_KERNEL_PROF(g_smiv->kPoolingHw,
                               curr_layer->num,
                               smiv_pooling_layer_hw,
                               current_inputs,
                               current_results,
                               g_smiv->umem,
                               g_smiv->spad0,
                               g_smiv->spad1,
                               partial_layer,
                               iteration_offset);

            iteration_offset += partial_input_size;
            current_inputs += partial_input_size;
            current_results += partial_output_size;
        }
    }

    dims_t output_dims =
            nchw_to_nhwc_dims(&curr_layer->outputs, DATA_ALIGNMENT);
    begin_profiling("convert_blocked_nhwc_to_nhwc", curr_layer->num);
    convert_blocked_nhwc_to_nchw(nhwc_outputs, 0, NUM_TEST_CASES, VECTOR_SIZE,
                                 output_dims, DATA_ALIGNMENT, results);
    end_profiling();
    free_smiv_work_cfg(&pool_cfgs);
    free_data_list(nhwc_inputs);
    free_data_list(nhwc_outputs);
}
