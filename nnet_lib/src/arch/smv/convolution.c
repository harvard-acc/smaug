#include <string.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "core/smv/params.h"
#include "core/smv/smv.h"
#include "utility/data_layout_conversion.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _conv_tile {
    int input_dims[5];
    int input_pad;
    int num_ofmaps;
} conv_tile;

typedef struct _conv_tiling_cfg {
    conv_tile* tiles;
    int num_tiles;
} conv_tiling_cfg;

typedef struct _smv_convolution_options {
    int img;
    int kern_start;
    int kern_end;
    int total_tile_ofmaps;
    bool use_pipelined_dma;
} smv_convolution_options;

void init_conv_tiling_cfg(conv_tiling_cfg* cfg, int num_tiles) {
    cfg->num_tiles = num_tiles;
    cfg->tiles = (conv_tile*)malloc(sizeof(conv_tile) * num_tiles);
    for (int i = 0; i < num_tiles; i++) {
        memset(&cfg->tiles[i], 0, sizeof(conv_tile));
    }
}

void free_conv_tiling_cfg(conv_tiling_cfg* cfg) {
    free(cfg->tiles);
}

void print_conv_tiling_cfg(conv_tiling_cfg* cfg) {
    for (int i = 0; i < cfg->num_tiles; i++) {
        INFO_MSG("Tile %d:\n"
                 "  OFMaps: %d\n"
                 "  IFMap size: %d, %d, %d, %d, %d\n",
                 i,
                 cfg->tiles[i].num_ofmaps,
                 cfg->tiles[i].input_dims[0],
                 cfg->tiles[i].input_dims[1],
                 cfg->tiles[i].input_dims[2],
                 cfg->tiles[i].input_dims[3],
                 cfg->tiles[i].input_dims[4]);
    }
}

static void smv_convolution_layer_hw_impl(float* host_activations,
                                          float* host_weights,
                                          float* host_results,
                                          float* local_activations,
                                          float* local_weights,
                                          float* local_results,
                                          layer_t curr_layer,
                                          smv_convolution_options* options) {
    int input_height = curr_layer.inputs.height;
    int input_rows = curr_layer.inputs.rows;
    int input_cols = curr_layer.inputs.cols;
    int input_pad = curr_layer.inputs.align_pad;
    ARRAY_4D(float, _a, host_activations, input_height + input_pad, input_rows,
             input_cols);
    // DMA all the weights that we can fit in the current tile (which is
    // specified by this tile's outputs.height).
    // We should only DMA part of the weights.
    int single_weights_elems =
            curr_layer.weights.rows * curr_layer.weights.cols *
            (curr_layer.weights.height + curr_layer.weights.align_pad);
    int num_weights = options->total_tile_ofmaps * single_weights_elems;
    if (curr_layer.weights_req != IO_NONE) {
        setReadyBits(local_weights, num_weights * sizeof(float), 0);
        dma_load_wrapper(local_weights,
                         host_weights,
                         num_weights * sizeof(float),
                         options->use_pipelined_dma);
    }
    if (curr_layer.input_req == IO_DMA || curr_layer.input_req == IO_ACP) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                curr_layer.inputs.rows * curr_layer.inputs.cols *
                (curr_layer.inputs.height + curr_layer.inputs.align_pad);
        if (curr_layer.input_req == IO_DMA) {
            setReadyBits(
                    local_activations, num_input_pixels * sizeof(float), 0);
            dma_load_wrapper(local_activations,
                             &_a[options->img][0][0][0],
                             num_input_pixels * sizeof(float),
                             options->use_pipelined_dma);
        } else {
            int source_offset = &_a[options->img][0][0][0] - &_a[0][0][0][0];
            coherentLoad64(local_activations, &_a[options->img][0][0][0],
                           num_input_pixels * sizeof(float), 0, source_offset);
        }
    }

    // We will invoke the hardware multiple times, but we don't DMA every time.
    // So, we need to read weights from and write outputs to the kern_start'th
    // output channel.
    convolution3d_smv(local_activations, local_weights, curr_layer,
                      options->kern_start, local_results);

    // Run the activation function in-place if applicable on the ofmaps we
    // generated.
    int ofmap_2d_elems =
            curr_layer.outputs.rows *
            (curr_layer.outputs.cols + curr_layer.outputs.align_pad);
    int num_output_pixels =
            (options->kern_end - options->kern_start) * ofmap_2d_elems;
    int start_output_pixel = options->kern_start * ofmap_2d_elems;
    smv_activation_fun_fxp(local_results, NUM_TEST_CASES, num_output_pixels,
                           start_output_pixel, curr_layer.activation);
    if (curr_layer.output_req == IO_DMA) {
        dma_store_wrapper(&host_results[start_output_pixel],
                          &local_results[start_output_pixel],
                          num_output_pixels * sizeof(float),
                          options->use_pipelined_dma);
    } else if (curr_layer.output_req == IO_ACP) {
        int output_offset = start_output_pixel / VECTOR_SIZE / 2;
        coherentStore64(host_results, local_results,
                        num_output_pixels * sizeof(float), output_offset,
                        output_offset);
    }
}

static void smv_convolution_layer_hw(float* dma_activations,
                                     float* dma_weights,
                                     float* dma_results,
                                     float* cache_activations,
                                     float* cache_weights,
                                     float* cache_results,
                                     float* acp_activations,
                                     float* acp_weights,
                                     float* acp_results,
                                     float* umem,
                                     float* spad0,
                                     float* spad1,
                                     layer_t curr_layer,
                                     access_config* access_config,
                                     smv_convolution_options* options) {
    // We can use ACP or caches for outputs only, or for outputs AND inputs,
    // but not inputs only. We also do not currently support mixing ACP and
    // cache for the inputs/outputs.
    if (access_config->outputs == _ACP) {
        if (access_config->inputs == _ACP) {
            smv_convolution_layer_hw_impl(acp_activations, dma_weights,
                                          acp_results, umem, spad0, spad1,
                                          curr_layer, options);
        } else {
            // If the input mechanism is Cache, then it is ignored, and we
            // fallback to DMA.
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            smv_convolution_layer_hw_impl(dma_activations, dma_weights,
                                          acp_results, umem, spad0, spad1,
                                          curr_layer, options);
        }
    } else if (access_config->outputs == _Cache) {
        if (access_config->inputs == _Cache) {
            smv_convolution_layer_hw_impl(dma_activations, dma_weights,
                                          dma_results, cache_activations, spad0,
                                          cache_results, curr_layer, options);
        } else {
            // If the input mechanism is ACP, then it is ignored, and we
            // fallback to DMA.
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            smv_convolution_layer_hw_impl(
                    dma_activations, dma_weights, dma_results, umem, spad0,
                    cache_results, curr_layer, options);
        }
    } else {
        ASSERT((access_config->inputs == _DmaOrLocal &&
                access_config->outputs == _DmaOrLocal) &&
               "IO requirements are inconsistent with DMA fallback!");
        smv_convolution_layer_hw_impl(dma_activations, dma_weights, dma_results,
                                      umem, spad0, spad1, curr_layer, options);
    }
}

static layer_t create_partial_layer_from_tile(layer_t* full_layer,
                                              conv_tile* tile) {
    layer_t partial_layer = *full_layer;
    partial_layer.outputs.height = tile->num_ofmaps;
    partial_layer.weights.height = tile->input_dims[0];
    partial_layer.inputs.align_pad = tile->input_pad;
    partial_layer.weights.align_pad = tile->input_pad;
    // Output padding does not need to be recalculated because the kernel
    // produces data in NCHW format.
    return partial_layer;
}

static conv_tiling_cfg convolution_divide_work(layer_t* curr_layer) {
    // Ensure that all the tiling is done using NHWC padding on the inputs (not
    // the outputs - they get written in NCHW!).
    layer_t curr_layer_nhwc_padded = *curr_layer;
    curr_layer_nhwc_padded.weights.align_pad =
            calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);
    curr_layer_nhwc_padded.inputs.align_pad =
            calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);

    conv_tiling_cfg cfg;
    int total_input_bytes =
            (get_dims_size(&curr_layer_nhwc_padded.inputs) * sizeof(float)) /
            NUM_TEST_CASES;

    if (total_input_bytes > SMV_UMEM_SIZE) {
        printf("A single input image exceeds the capacity of the UMEM, which "
               "is not supported!\n");
        assert(false);
    }

    const int output_2d_size = curr_layer_nhwc_padded.outputs.rows *
                               (curr_layer_nhwc_padded.outputs.cols +
                                curr_layer_nhwc_padded.outputs.align_pad) *
                               sizeof(float);
    if (output_2d_size > SMV_SPAD_SIZE) {
        fprintf(stderr,
                "A single output channel doesn't fit on the scratchpad! We "
                "don't support this mode of tiling yet!\n");
        assert(false);
    }

    // Divide up the work over output channels.
    // The number of output feature maps we can support at once is determined
    // by how many weights and output feature maps can fit into the two
    // scratchpads.
    const int single_kernel_size =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.weights) * sizeof(float);
    const int max_kernels_per_iter = SMV_SPAD_SIZE / single_kernel_size;
    const int max_ofmaps_per_iter = SMV_SPAD_SIZE / output_2d_size;
    const int num_ofmaps_per_iter =
            min2(max_kernels_per_iter, max_ofmaps_per_iter);
    const int num_iters =
            ceil(((float)curr_layer_nhwc_padded.outputs.height) / num_ofmaps_per_iter);
    init_conv_tiling_cfg(&cfg, num_iters);
    int remaining_ofmaps = curr_layer_nhwc_padded.outputs.height;
    for (int i = 0; i < num_iters; i++) {
        cfg.tiles[i].input_dims[0] = curr_layer_nhwc_padded.inputs.height;
        cfg.tiles[i].input_dims[1] = curr_layer_nhwc_padded.inputs.cols;
        cfg.tiles[i].input_dims[2] = curr_layer_nhwc_padded.inputs.rows;
        cfg.tiles[i].input_dims[3] = NUM_TEST_CASES;
        cfg.tiles[i].input_dims[4] = 1;
        cfg.tiles[i].input_pad =
                calc_padding(cfg.tiles[0].input_dims[0], DATA_ALIGNMENT);
        cfg.tiles[i].num_ofmaps = min2(remaining_ofmaps, num_ofmaps_per_iter);
        remaining_ofmaps -= num_ofmaps_per_iter;
    }
    return cfg;
}

void smv_standard_convolution_layer_impl(float* host_activations,
                                         float* host_weights,
                                         layer_t* layers,
                                         int lnum,
                                         float* host_result,
                                         smv_global* g_smv,
                                         device_t* device,
                                         sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    const int result_height = curr_layer.outputs.height;
    const int result_rows = curr_layer.outputs.rows;
    const int result_cols = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int num_kerns = curr_layer.outputs.height;
    const int input_height = curr_layer.inputs.height;
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    const int result_2d_size = result_rows * (result_cols + result_pad);
    const int activations_size = INPUT_BYTES(layers, lnum);
    const int weights_size = WEIGHT_BYTES(layers, lnum);
    float* nhwc_activations = NULL;
    begin_profiling("convert_nchw_to_nhwc", lnum);
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            host_activations, NUM_TEST_CASES, curr_layer.inputs, DATA_ALIGNMENT,
            &nhwc_activations);
    end_profiling();
    MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                       get_host_inputs_var_name(curr_layer.input_req),
                       nhwc_activations,
                       get_dims_size(&activations_nhwc) * sizeof(float));

    ARRAY_4D(float, _result, host_result, result_height, result_rows,
             result_cols + result_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);

    conv_tiling_cfg tiling = convolution_divide_work(&curr_layer);
    print_conv_tiling_cfg(&tiling);

    bool do_hw_activation = device->use_hw_activation_func &&
                            smiv_is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    bool use_pipelined_dma = device->use_pipelined_dma;
    if (curr_layer.input_req == IO_DMA) {
        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        flush_cache_range(host_activations, activations_size / sizeof(float));
        flush_cache_range(host_weights, weights_size / sizeof(float));
        end_profiling();
    }
    if (do_hw_activation || curr_layer.output_req == IO_DMA) {
        // Flush cache lines for temporary results.
        begin_ignored_profiling(lnum);
        flush_cache_range(host_result, result_2d_size * result_height);
        end_profiling();
    }

    int sampled_inner_iters = sampling_param->smv_conv_inner_iters;
    // A value of 0 means do not sample, so set this value to the actual number
    // of inner iterations, if there are any.
    if (sampled_inner_iters == 0)
        sampled_inner_iters = max2(0, num_kerns - 2);
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        int kern_start = 0;
        for (int t = 0; t < tiling.num_tiles; t++) {
            conv_tile* tile = &tiling.tiles[t];
            layer_t partial_layer =
                    create_partial_layer_from_tile(&curr_layer, tile);
            if (t != 0 && curr_layer.input_req == IO_DMA)
                partial_layer.input_req = IO_NONE;

            // Set up the results buffer and mappings.
            float* result_loc = &_result[img][kern_start][0][0];
            int result_size = result_2d_size * tile->num_ofmaps * sizeof(float);
            MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                               get_host_results_var_name(curr_layer.output_req),
                               result_loc, result_size);

            // Convert weights to NHWC and set up mappings.
            float* nhwc_weights = NULL;
            begin_profiling("convert_nchw_to_nhwc", lnum);
            dims_t weights_nhwc = convert_nchw_to_nhwc(
                    &_kernels[kern_start][0][0][0], tile->num_ofmaps,
                    curr_layer.weights, DATA_ALIGNMENT, &nhwc_weights);
            end_profiling();
            MAP_ARRAY_TO_ACCEL(
                    g_smv->kConvolutionHw,
                    get_host_weights_var_name(IO_DMA),
                    nhwc_weights,
                    num_kerns * get_dims_size(&weights_nhwc) * sizeof(float));

            int num_hw_iters = ceil((float)tile->num_ofmaps / NUM_PE_INSTS);
            int inner_iters_executed = 0;

            // Sampling operates on iterations of the following loop over
            // num_hw_iters. We always execute the first and last iterations,
            // because those iterations are responsible for handling data
            // movement.  Between those two iterations, we only run up to
            // sampled_inner_iter iterations. The remaining iterations have
            // their outputs set to zero.
            begin_profiling("standard_convolution_layer_smv_impl_tile", lnum);
            bool is_sampled = (sampled_inner_iters > 0 &&
                               sampled_inner_iters < num_hw_iters - 2);
            if (is_sampled) {
                set_profiling_type_sampled(
                        sampled_inner_iters + 2, num_hw_iters);
            }
            for (int iter = 0; iter < num_hw_iters; iter++) {
                bool is_last_iter = (iter == num_hw_iters - 1);
                smv_convolution_options options;
                options.img = img;
                // This kern_start is with respect to the current set of output
                // fmaps in the tile.
                options.kern_start = iter * NUM_PE_INSTS;
                int num_kerns_this_iter =
                        min2(tile->num_ofmaps - options.kern_start,
                             (int)NUM_PE_INSTS);
                options.kern_end = options.kern_start + num_kerns_this_iter;
                // This is required to DMA the correct number of weights and
                // outputs back from the accelerator at the beginning and end.
                options.total_tile_ofmaps = tile->num_ofmaps;
                options.use_pipelined_dma = use_pipelined_dma;

                if (iter != 0 && !is_last_iter &&
                    inner_iters_executed >= sampled_inner_iters) {
                    // Skip this iteration and zero out the result.
                    memset(&_result[img][kern_start + options.kern_start][0][0],
                           0, result_2d_size * num_kerns_this_iter);
                    continue;
                }

                // Only copy weights and inputs on the first iteration.
                if (iter > 0) {
                    if (partial_layer.input_req == IO_DMA ||
                        partial_layer.input_req == IO_ACP)
                        partial_layer.input_req = IO_NONE;
                    if (partial_layer.weights_req == IO_DMA ||
                        partial_layer.weights_req == IO_ACP)
                        partial_layer.weights_req = IO_NONE;
                }
                // Only run the activation function on the last iteration.
                partial_layer.activation = (do_hw_activation)
                                                   ? curr_layer.activation
                                                   : NO_ACTIVATION;
                access_config access_cfg =
                        layer_to_access_config(&partial_layer);
                INVOKE_KERNEL_PROF(
                        g_smv->kConvolutionHw, lnum, smv_convolution_layer_hw,
                        nhwc_activations, nhwc_weights, result_loc,  // DMA
                        nhwc_activations, nhwc_weights, result_loc,  // Cache
                        nhwc_activations, nhwc_weights, result_loc,  // ACP
                        g_smv->umem, g_smv->spad0, g_smv->spad1, partial_layer,
                        &access_cfg, &options);

                if (iter != 0 && !is_last_iter) {
                    inner_iters_executed++;
                }
            }
            free(nhwc_weights);
            kern_start += tile->num_ofmaps;
            end_profiling();
        }
    }
    free(nhwc_activations);
    free_conv_tiling_cfg(&tiling);
}
