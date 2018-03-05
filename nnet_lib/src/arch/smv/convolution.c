#include <string.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
#include "core/smv/params.h"
#include "core/smv/smv.h"
#include "core/ref/activation_functions.h"
#include "utility/data_layout_conversion.h"
#include "utility/fp16_utils.h"
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

static void smv_convolution_layer_hw_impl(packed_fp16* host_activations,
                                          packed_fp16* host_weights,
                                          packed_fp16* host_results,
                                          float* local_activations,
                                          float* local_weights,
                                          float* local_results,
                                          layer_t curr_layer,
                                          smv_convolution_options* options) {
    int input_height = curr_layer.inputs.height;
    int input_rows = curr_layer.inputs.rows;
    int input_cols = curr_layer.inputs.cols;
    int input_pad = curr_layer.inputs.align_pad;
    ARRAY_4D(packed_fp16, _a, host_activations, input_rows, input_cols,
             input_height + input_pad);
    // DMA all the weights that we can fit in the current tile (which is
    // specified by this tile's outputs.height).
    // We should only DMA part of the weights.
    int single_weights_elems =
            curr_layer.weights.rows * curr_layer.weights.cols *
            (curr_layer.weights.height + curr_layer.weights.align_pad);
    int num_weights = options->total_tile_ofmaps * single_weights_elems;
    if (curr_layer.weights_req != IO_NONE) {
        setReadyBits(local_weights, num_weights * sizeof(*local_results), 0);
        dma_load_and_unpack_fp16(local_weights, host_weights, num_weights, 0, 0);
    }
    if (curr_layer.input_req == IO_DMA || curr_layer.input_req == IO_ACP) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                curr_layer.inputs.rows * curr_layer.inputs.cols *
                (curr_layer.inputs.height + curr_layer.inputs.align_pad);
        if (curr_layer.input_req == IO_DMA) {
            setReadyBits(local_activations,
                         num_input_pixels * sizeof(*local_activations), 0);
            dma_load_and_unpack_fp16(local_activations,
                                     &_a[options->img][0][0][0],
                                     num_input_pixels, 0, 0);
        } else {
            int source_offset = &_a[options->img][0][0][0] - &_a[0][0][0][0];
            acp_load_and_unpack_fp16(local_activations,
                                     &_a[options->img][0][0][0],
                                     num_input_pixels, 0, source_offset);
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
        dma_pack_and_store_fp16(host_results,
                                local_results,
                                num_output_pixels,
                                start_output_pixel,
                                start_output_pixel);
    } else if (curr_layer.output_req == IO_ACP) {
        acp_pack_and_store_fp16(host_results,
                                local_results,
                                num_output_pixels,
                                start_output_pixel,
                                start_output_pixel);
    }
}

static void smv_convolution_layer_hw(packed_fp16* dma_activations,
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
                                     layer_t curr_layer,
                                     access_config* access_config,
                                     smv_convolution_options* options) {
    // XXX: With half-precision data storage, we can't directly access an L1
    // cache, since we can't compute on half precision data without calling the
    // conversion functions on every access. So to prevent such a situation
    // from arising, just disable use of local caches.
    bool use_acp_outputs =
            access_config->outputs == _ACP || access_config->outputs == _Cache;
    bool use_acp_inputs =
            access_config->inputs == _ACP || access_config->inputs == _Cache;
    if (use_acp_outputs) {
        curr_layer.output_req = IO_ACP;
        if (use_acp_inputs) {
            curr_layer.input_req = IO_ACP;
            smv_convolution_layer_hw_impl(acp_activations, dma_weights,
                                          acp_results, umem, spad0, spad1,
                                          curr_layer, options);
        } else {
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            smv_convolution_layer_hw_impl(dma_activations, dma_weights,
                                          acp_results, umem, spad0, spad1,
                                          curr_layer, options);
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

void smv_standard_convolution_layer_impl(data_list* host_activations,
                                         data_list* host_weights,
                                         layer_t* layers,
                                         int lnum,
                                         data_list* host_results,
                                         smv_global* g_smv,
                                         device_t* device,
                                         sampling_param_t* sampling_param) {
    require_data_type(host_weights, 0, UncompressedHalfPrecision);
    require_data_type(host_activations, 0, UncompressedHalfPrecision);

    layer_t curr_layer = layers[lnum];
    const int result_height = curr_layer.outputs.height;
    const int result_rows = curr_layer.outputs.rows;
    const int result_cols = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int num_kerns = curr_layer.outputs.height;
    const int input_height = curr_layer.inputs.height;
    const int k_rows = curr_layer.weights.rows;
    const int k_cols = curr_layer.weights.cols;
    const int result_2d_size = result_rows * (result_cols + result_pad);

    data_list* nhwc_activations = init_data_list(1);
    begin_profiling("convert_nchw_to_nhwc", lnum);
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            host_activations, 0, NUM_TEST_CASES, curr_layer.inputs,
            DATA_ALIGNMENT, nhwc_activations);
    end_profiling();
    packed_fp16* activations_loc = nhwc_activations->data[0].dense_hp->d;
    // TODO: Add metadata to indicate the size of elements contained inside
    // DataFormat.
    MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                       get_host_inputs_var_name(curr_layer.input_req),
                       activations_loc,
                       get_dims_size(&activations_nhwc) * sizeof(float16));

    // XXX: host_weights arrives in NHWC format, but layer.weights is still in
    // NCHW dimension format.
    dims_t nhwc_weights_dims =
            nchw_to_nhwc_dims(&curr_layer.weights, DATA_ALIGNMENT);
    // host_weights is half-precision, so it only occupies half the space that
    // the logical dimensions would indicate.
    ARRAY_4D(packed_fp16, _kernels, host_weights->data[0].dense_hp->d, k_rows,
             k_cols, (input_height + nhwc_weights_dims.align_pad) / 2);
    ARRAY_4D(packed_fp16, _result, host_results->data[0].dense_hp->d,
             result_height, result_rows, (result_cols + result_pad) / 2);

    conv_tiling_cfg tiling = convolution_divide_work(&curr_layer);
    print_conv_tiling_cfg(&tiling);

    bool do_hw_activation = device->use_hw_activation_func &&
                            smiv_is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    bool use_pipelined_dma = device->use_pipelined_dma;
    bool use_pipelined_activation = device->use_pipelined_activation_func;
    if (curr_layer.input_req == IO_DMA) {
        begin_ignored_profiling(lnum);
        flush_cache_range(
                host_activations->data[0].dense_hp->d,
                host_activations->data[0].dense_hp->size * sizeof(float16));
        flush_cache_range(
                host_weights->data[0].dense_hp->d,
                host_weights->data[0].dense_hp->size * sizeof(float16));
        end_profiling();
    }
    if (do_hw_activation || curr_layer.output_req == IO_DMA) {
        begin_ignored_profiling(lnum);
        flush_cache_range(
                host_results->data[0].dense_hp->d,
                host_results->data[0].dense_hp->size * sizeof(float16));
        end_profiling();
    }

    int sampled_inner_iters = sampling_param->smv_conv_inner_iters;
    // A value of 0 means do not sample, so set this value to the actual number
    // of inner iterations, if there are any.
    if (sampled_inner_iters == 0)
        sampled_inner_iters = max2(0, num_kerns - 2);

    volatile int finish_flag;
#ifdef GEM5_HARNESS
    finish_flag = NOT_COMPLETED;
#else
    finish_flag = 0;
#endif
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        int kern_start = 0;
        for (int t = 0; t < tiling.num_tiles; t++) {
            conv_tile* tile = &tiling.tiles[t];
            layer_t partial_layer =
                    create_partial_layer_from_tile(&curr_layer, tile);
            if (t != 0 && curr_layer.input_req == IO_DMA)
                partial_layer.input_req = IO_NONE;

            // Set up the results buffer and mappings.
            packed_fp16* result_loc = &_result[img][kern_start][0][0];
            int result_size = result_2d_size * tile->num_ofmaps * sizeof(float16);
            MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                               get_host_results_var_name(curr_layer.output_req),
                               result_loc, result_size);

            // Convert weights to NHWC and set up mappings.
            packed_fp16* weights_loc = &_kernels[kern_start][0][0][0];
            MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                               get_host_weights_var_name(IO_DMA),
                               weights_loc,
                               num_kerns * get_dims_size(&nhwc_weights_dims) *
                                       sizeof(float16));

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
            // Used for the pipelined activation mechanism. The current iteration
            // will do activation function on the results produced by previous
            // iteration.
            int prev_iter_num_kerns = 0;
            int prev_kern_start = 0;
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
                if (!use_pipelined_activation) {
                    INVOKE_KERNEL_PROF(
                            g_smv->kConvolutionHw, lnum,
                            smv_convolution_layer_hw,
                            activations_loc, weights_loc, result_loc,  // DMA
                            activations_loc, weights_loc, result_loc,  // Cache
                            activations_loc, weights_loc, result_loc,  // ACP
                            g_smv->umem, g_smv->spad0, g_smv->spad1,
                            partial_layer, &access_cfg, &options);

                } else {
                    // If the previous iteration has finished, start doing
                    // activation functions and invoke the next iteration
                    // simultaneously. Otherwise, wait until the previous iteration
                    // finishes.
                    #ifdef GEM5_HARNESS
                    while (iter != 0 && finish_flag == NOT_COMPLETED)
                        ;
                    #endif
                    if (iter != 0)
                        end_profiling();
                    begin_profiling("smv_convolution_layer_hw_activation_func", lnum);
                    INVOKE_KERNEL_NOBLOCK(
                            g_smv->kConvolutionHw, &finish_flag,
                            smv_convolution_layer_hw,
                            activations_loc, weights_loc, result_loc,  // DMA
                            activations_loc, weights_loc, result_loc,  // Cache
                            activations_loc, weights_loc, result_loc,  // ACP
                            g_smv->umem, g_smv->spad0, g_smv->spad1,
                            partial_layer, &access_cfg, &options);
                    if (iter != 0 && !do_hw_activation) {
                        begin_profiling(
                                ACTIVATION_TYPE_STR(curr_layer.activation),
                                lnum);
                        activation_fun_simd128(
                                &_result[img][prev_kern_start][0][0],
                                1,
                                result_2d_size * prev_iter_num_kerns,
                                curr_layer.activation,
                                &_result[img][prev_kern_start][0][0]);
                        end_profiling();
                    }
                    prev_iter_num_kerns = num_kerns_this_iter;
                    prev_kern_start = kern_start + options.kern_start;
                }

                if (iter != 0 && !is_last_iter) {
                    inner_iters_executed++;
                }
            }
            // We need to do activation function for the last iteration of the
            // tile.
            if (use_pipelined_activation && !do_hw_activation) {
                #ifdef GEM5_HARNESS
                while (finish_flag == NOT_COMPLETED)
                    ;
                #endif
                end_profiling();
                begin_profiling(
                        ACTIVATION_TYPE_STR(curr_layer.activation), lnum);
                activation_fun_simd128(&_result[img][prev_kern_start][0][0],
                                       1,
                                       result_2d_size * prev_iter_num_kerns,
                                       curr_layer.activation,
                                       &_result[img][prev_kern_start][0][0]);
                end_profiling();
            }

            kern_start += tile->num_ofmaps;
            end_profiling();
        }
    }
    free_data_list(nhwc_activations);
    free_conv_tiling_cfg(&tiling);
}
