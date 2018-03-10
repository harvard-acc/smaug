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

typedef struct _conv_output_tile {
    int output_dims[5];
    int output_pad;
    int num_ofmaps;
} conv_output_tile;

typedef struct _conv_input_tile {
    int input_dims[5];
    int input_pad;
    padding pad;
    conv_output_tile* output_tiles;
    int num_output_tiles;
} conv_input_tile;

typedef struct _conv_tiling_cfg {
    conv_input_tile* input_tiles;
    int num_input_tiles;
} conv_tiling_cfg;

typedef struct _smv_convolution_options {
    int img;
    int kern_start;
    int kern_end;
    int total_tile_ofmaps;
    bool use_pipelined_dma;
} smv_convolution_options;

void free_conv_tiling_cfg(conv_tiling_cfg* cfg) {
    for (int i = 0; i < cfg->num_input_tiles; i++) {
        free(cfg->input_tiles[i].output_tiles);
    }
    free(cfg->input_tiles);
}

void print_conv_tiling_cfg(conv_tiling_cfg* cfg, int lnum) {
    INFO_MSG("\nTiling info for layer %d\n", lnum);
    for (int i = 0; i < cfg->num_input_tiles; i++) {
        conv_input_tile* input_tile =
                &cfg->input_tiles[i];
        INFO_MSG("Input tile %d\n"
                 "  IFMap size: %d, %d, %d, %d, %d\n"
                 "  zero padding: %d, %d, %d, %d\n"
                 "  input pad: %d\n",
                 i,
                 input_tile->input_dims[0],
                 input_tile->input_dims[1],
                 input_tile->input_dims[2],
                 input_tile->input_dims[3],
                 input_tile->input_dims[4],
                 input_tile->pad.top,
                 input_tile->pad.bottom,
                 input_tile->pad.left,
                 input_tile->pad.right,
                 input_tile->input_pad);
        for (int j = 0; j < input_tile->num_output_tiles; j++) {
            conv_output_tile* output_tile =
                    &input_tile->output_tiles[j];
            INFO_MSG("    Output tile %d:\n"
                     "      OFMaps: %d\n"
                     "      OFMap size: %d, %d, %d, %d, %d\n"
                     "      output pad %d\n",
                     j,
                     output_tile->num_ofmaps,
                     output_tile->output_dims[0],
                     output_tile->output_dims[1],
                     output_tile->output_dims[2],
                     output_tile->output_dims[3],
                     output_tile->output_dims[4],
                     output_tile->output_pad);
        }
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
                                              conv_input_tile* input_tile,
                                              conv_output_tile* output_tile) {
    layer_t partial_layer = *full_layer;
    partial_layer.inputs.rows = input_tile->input_dims[2];
    partial_layer.inputs.cols = input_tile->input_dims[1];
    partial_layer.inputs.height = input_tile->input_dims[0];
    partial_layer.inputs.align_pad = input_tile->input_pad;
    partial_layer.outputs.rows = output_tile->output_dims[2];
    partial_layer.outputs.cols = output_tile->output_dims[1];
    partial_layer.outputs.height = output_tile->output_dims[0];
    partial_layer.outputs.align_pad = output_tile->output_pad;
    partial_layer.weights.height = input_tile->input_dims[0];
    partial_layer.weights.align_pad = input_tile->input_pad;
    partial_layer.pad = input_tile->pad;
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
    const int total_input_bytes =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.inputs) * sizeof(float);
    bool need_input_tiling = false;
    int num_input_iters = 1;
    int halo_rows = 0;
    int num_rows_per_iter = curr_layer_nhwc_padded.inputs.rows;
    int output_2d_rows = curr_layer_nhwc_padded.outputs.rows;
    int first_output_2d_rows = curr_layer_nhwc_padded.outputs.rows;
    int last_output_2d_rows = curr_layer_nhwc_padded.outputs.rows;
    int output_2d_size = curr_layer_nhwc_padded.outputs.rows *
                         (curr_layer_nhwc_padded.outputs.cols +
                          curr_layer_nhwc_padded.outputs.align_pad) *
                         sizeof(float);
    int first_output_2d_size = output_2d_size;
    int last_output_2d_size = output_2d_size;

    // If the input can't fit on the UMEM, then we need to do input tiling.
    // The input is tiled based on a strip mining mechanism, the smallest tile
    // is of (K * W * H) layout format, where K is the kernel's length, W is
    // is input's width, H is input's height.
    if (total_input_bytes > SMV_UMEM_SIZE) {
        need_input_tiling = true;
        const int single_strip_size =
                curr_layer_nhwc_padded.weights.rows *
                curr_layer_nhwc_padded.inputs.cols *
                (curr_layer_nhwc_padded.inputs.height +
                 curr_layer_nhwc_padded.inputs.align_pad) *
                sizeof(float);
        if (single_strip_size > SMV_UMEM_SIZE) {
            printf("A single strip of the input image exceeds the capacity of "
                   "the UMEM, which is not supported!\n");
            assert(false);
        }

        // Divide up the work over input strips.
        const int max_strip_per_iter = SMV_UMEM_SIZE / single_strip_size;
        halo_rows = curr_layer_nhwc_padded.weights.rows -
                    curr_layer_nhwc_padded.field_stride;
        num_rows_per_iter =
                max_strip_per_iter * curr_layer_nhwc_padded.weights.rows;
        num_input_iters =
                ceil((float)(curr_layer_nhwc_padded.inputs.rows - halo_rows) /
                     (curr_layer_nhwc_padded.weights.rows * max_strip_per_iter -
                      halo_rows));
        const int num_rows_last_iter =
                curr_layer_nhwc_padded.inputs.rows -
                (num_rows_per_iter - halo_rows) * (num_input_iters - 1);
        output_2d_rows =
                (num_rows_per_iter - curr_layer_nhwc_padded.weights.rows) /
                        curr_layer_nhwc_padded.field_stride +
                1;
        output_2d_size =
                output_2d_rows *
                (curr_layer_nhwc_padded.outputs.cols +
                 curr_layer_nhwc_padded.outputs.align_pad) *
                sizeof(float);
        first_output_2d_rows =
                (num_rows_per_iter - curr_layer_nhwc_padded.weights.rows +
                 curr_layer_nhwc_padded.pad.top) /
                        curr_layer_nhwc_padded.field_stride +
                1;
        first_output_2d_size = first_output_2d_rows *
                               (curr_layer_nhwc_padded.outputs.cols +
                                curr_layer_nhwc_padded.outputs.align_pad) *
                               sizeof(float);
        last_output_2d_rows =
                (num_rows_last_iter - curr_layer_nhwc_padded.weights.rows +
                 curr_layer_nhwc_padded.pad.bottom) /
                        curr_layer_nhwc_padded.field_stride +
                1;
        last_output_2d_size =
                last_output_2d_rows *
                (curr_layer_nhwc_padded.outputs.cols +
                 curr_layer_nhwc_padded.outputs.align_pad) *
                sizeof(float);
    }

    if (output_2d_size > SMV_SPAD_SIZE) {
        fprintf(stderr,
                "A single output channel of the input tile"
                "doesn't fit on the scratchpad! We "
                "don't support this mode of tiling yet!\n");
        assert(false);
    }

    // Divide up the work over output channels.
    // The number of output feature maps we can support at once is determined
    // by how many weights and output feature maps can fit into the two
    // scratchpads.
    const int single_kernel_size =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.weights) * sizeof(float);
    const int max_kernels_per_iter =
                    SMV_SPAD_SIZE / single_kernel_size;
    const int max_ofmaps_per_iter = SMV_SPAD_SIZE / output_2d_size;
    const int num_ofmaps_per_iter =
            min2(max_kernels_per_iter, max_ofmaps_per_iter);
    const int num_output_iters =
            ceil(((float)curr_layer_nhwc_padded.outputs.height) / num_ofmaps_per_iter);

    const int max_ofmaps_first_iter = SMV_SPAD_SIZE / first_output_2d_size;
    const int num_ofmaps_first_iter =
            min2(max_kernels_per_iter, max_ofmaps_first_iter);
    const int num_output_first_iter =
            ceil(((float)curr_layer_nhwc_padded.outputs.height) /
                 num_ofmaps_first_iter);

    const int max_ofmaps_last_iter = SMV_SPAD_SIZE / last_output_2d_size;
    const int num_ofmaps_last_iter =
            min2(max_kernels_per_iter, max_ofmaps_last_iter);
    const int num_output_last_iter =
            ceil(((float)curr_layer_nhwc_padded.outputs.height) /
                 num_ofmaps_last_iter);

    // Create tiling configurations.
    cfg.num_input_tiles = num_input_iters;
    cfg.input_tiles =
            (conv_input_tile*)malloc(sizeof(conv_input_tile) * num_input_iters);
    int remaining_input_rows = curr_layer_nhwc_padded.inputs.rows;
    for (int i = 0; i < cfg.num_input_tiles; i++) {
        bool first_input_tile = (i == 0);
        bool last_input_tile = (i == cfg.num_input_tiles - 1);
        // The last input tile has a different number of output tiles.
        conv_input_tile* input_tile = &cfg.input_tiles[i];
        input_tile->num_output_tiles =
                first_input_tile ? num_output_first_iter
                                 : last_input_tile ? num_output_last_iter
                                                   : num_output_iters;
        input_tile->output_tiles = (conv_output_tile*)malloc(
                sizeof(conv_output_tile) * cfg.input_tiles[i].num_output_tiles);
        input_tile->input_dims[0] = curr_layer_nhwc_padded.inputs.height;
        input_tile->input_dims[1] = curr_layer_nhwc_padded.inputs.cols;
        input_tile->input_dims[2] =
                min2(remaining_input_rows, num_rows_per_iter);
        input_tile->input_dims[3] = NUM_TEST_CASES;
        input_tile->input_dims[4] = 1;
        input_tile->pad = curr_layer_nhwc_padded.pad;
        // If we have more than one input tile, we need to take care of zero
        // padding for all the input tiles. The first tile will have no bottom
        // padding, and the last tile will have no top padding. The rest will
        // have no top and bottom paddings.
        if (cfg.num_input_tiles > 1) {
            if (first_input_tile) {
                input_tile->pad.bottom = 0;
            } else if (last_input_tile) {
                input_tile->pad.top = 0;
            } else {
                input_tile->pad.top = 0;
                input_tile->pad.bottom = 0;
            }
        }
        input_tile->input_pad =
                calc_padding(input_tile->input_dims[0], DATA_ALIGNMENT);
        remaining_input_rows -= (num_rows_per_iter - halo_rows);
        int remaining_ofmaps = curr_layer_nhwc_padded.outputs.height;
        for (int j = 0; j < input_tile->num_output_tiles; j++) {
            conv_output_tile* output_tile = &input_tile->output_tiles[j];
            if (first_input_tile) {
                output_tile->num_ofmaps =
                        min2(remaining_ofmaps, num_ofmaps_first_iter);
            } else if (last_input_tile) {
                output_tile->num_ofmaps =
                        min2(remaining_ofmaps, num_ofmaps_last_iter);
            } else {
                output_tile->num_ofmaps =
                        min2(remaining_ofmaps, num_ofmaps_per_iter);
            }
            output_tile->output_dims[0] = output_tile->num_ofmaps;
            output_tile->output_dims[1] = curr_layer_nhwc_padded.outputs.cols;
            // Calculate the number of rows in the output of the tile.
            // NOTE: we need to do padding for the last input tile.
            if (first_input_tile) {
                output_tile->output_dims[2] = first_output_2d_rows;
            } else if (last_input_tile) {
                output_tile->output_dims[2] = last_output_2d_rows;
            } else {
                output_tile->output_dims[2] = output_2d_rows;
            }
            output_tile->output_dims[3] = NUM_TEST_CASES;
            output_tile->output_dims[4] = 1;
            output_tile->output_pad =
                    calc_padding(output_tile->output_dims[1], DATA_ALIGNMENT);
            remaining_ofmaps -= output_tile->num_ofmaps;
        }
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
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int k_rows = curr_layer.weights.rows;
    const int k_cols = curr_layer.weights.cols;

    data_list* nhwc_activations = init_data_list(1);
    begin_profiling("convert_nchw_to_nhwc", lnum);
    dims_t activations_nhwc = convert_nchw_to_nhwc(host_activations,
                                                   0,
                                                   NUM_TEST_CASES,
                                                   curr_layer.inputs,
                                                   DATA_ALIGNMENT,
                                                   nhwc_activations);
    end_profiling();
    packed_fp16* activations = nhwc_activations->data[0].dense_hp->d;
    ARRAY_4D(float16,
             _activations,
             activations,
             input_rows,
             input_cols,
             input_height + activations_nhwc.align_pad);
    // TODO: Add metadata to indicate the size of elements contained inside
    // DataFormat.
    MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                       get_host_inputs_var_name(curr_layer.input_req),
                       activations,
                       get_dims_size(&activations_nhwc) * sizeof(float16));

    // XXX: host_weights arrives in NHWC format, but layer.weights is still in
    // NCHW dimension format.
    dims_t nhwc_weights_dims =
            nchw_to_nhwc_dims(&curr_layer.weights, DATA_ALIGNMENT);
    // host_weights is half-precision, so it only occupies half the space that
    // the logical dimensions would indicate.
    ARRAY_4D(float16, _kernels, host_weights->data[0].dense_hp->d, k_rows,
             k_cols, input_height + nhwc_weights_dims.align_pad);
    ARRAY_4D(float16, _result, host_results->data[0].dense_hp->d,
             result_height, result_rows, result_cols + result_pad);

    conv_tiling_cfg tiling = convolution_divide_work(&curr_layer);
    print_conv_tiling_cfg(&tiling, lnum);

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

    int sampled_output_tiles = sampling_param->smv_conv_output_tiles;
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
    // Outermost loop for batching.
    int halo_rows = curr_layer.weights.rows - curr_layer.field_stride;
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        // Outer loop for input tiling. The input is tiled along rows, and the
        // output of an input tile is furtur tiled along output channels.
        // We need to track the start row number of the current input tile and
        // its result in the original input and result buffers, respectively.
        int input_row_start = 0;
        int result_row_start = 0;
        for (int it = 0; it < tiling.num_input_tiles; it++) {
            conv_input_tile* input_tile = &tiling.input_tiles[it];
            layer_t partial_layer;
            int kern_start = 0;
            int output_tiles_executed = 0;
            // Output tile sampling operates over the following loop. We always
            // execute the last output tile, because it has a different number
            // of output feature maps. We only run up to sampled_output_tiles
            // iterations.
            begin_profiling("standard_convolution_layer_smv_input_tile", lnum);
            bool is_output_tile_sampled =
                    (sampled_output_tiles > 0 &&
                     sampled_output_tiles < input_tile->num_output_tiles - 1);
            if (is_output_tile_sampled) {
                set_profiling_type_sampled(
                        sampled_output_tiles + 1, input_tile->num_output_tiles);
            }
            // Inner loop for output tiling of an input tile.
            for (int ot = 0; ot < input_tile->num_output_tiles; ot++) {
                bool is_last_output_tile = (ot == input_tile->num_output_tiles - 1);
                if (is_output_tile_sampled && !is_last_output_tile &&
                    output_tiles_executed >= sampled_output_tiles) {
                    continue;
                }
                begin_profiling(
                        "standard_convolution_layer_smv_output_tile", lnum);
                conv_output_tile* output_tile = &input_tile->output_tiles[ot];
                // NOTE: partial_layer's inputs are in NHWC format. So use
                // get_nhwc_dims_size() to get the input size instead of
                // get_dims_size().
                partial_layer = create_partial_layer_from_tile(
                        &curr_layer, input_tile, output_tile);
                if (ot != 0 && curr_layer.input_req == IO_DMA)
                    partial_layer.input_req = IO_NONE;

                // Set up input location and mappings.
                float16* activations_loc =
                        &_activations[img][input_row_start][0][0];
                MAP_ARRAY_TO_ACCEL(
                        g_smv->kConvolutionHw,
                        get_host_inputs_var_name(curr_layer.input_req),
                        activations_loc,
                        get_nhwc_dims_size(&partial_layer.inputs) *
                                sizeof(float16));

                // Set up the results buffer and mappings.
                int partial_result_2d_size = partial_layer.outputs.rows *
                                             (partial_layer.outputs.cols +
                                              partial_layer.outputs.align_pad);
                int partial_result_size = partial_result_2d_size *
                                          output_tile->num_ofmaps *
                                          sizeof(float16);
                packed_fp16* temp_result_buf =
                        (packed_fp16*)malloc_aligned(partial_result_size);
                memset(temp_result_buf, 0, partial_result_size);
                MAP_ARRAY_TO_ACCEL(
                        g_smv->kConvolutionHw,
                        get_host_results_var_name(curr_layer.output_req),
                        temp_result_buf,
                        partial_result_size);

                // Convert weights to NHWC and set up mappings.
                float16* weights_loc = &_kernels[kern_start][0][0][0];
                MAP_ARRAY_TO_ACCEL(g_smv->kConvolutionHw,
                                   get_host_weights_var_name(IO_DMA),
                                   weights_loc,
                                   num_kerns *
                                           get_dims_size(&nhwc_weights_dims) *
                                           sizeof(float16));

                int num_hw_iters =
                        ceil((float)output_tile->num_ofmaps / NUM_PE_INSTS);
                int inner_iters_executed = 0;

                // Sampling operates on iterations of the following loop over
                // num_hw_iters. We always execute the first and last
                // iterations, because those iterations are responsible for
                // handling data movement. Between those two iterations, we only
                // run up to sampled_inner_iter iterations. The remaining
                // iterations have their outputs set to zero.
                begin_profiling(
                        "standard_convolution_layer_smv_impl_tile", lnum);
                bool is_inner_iter_sampled = (sampled_inner_iters > 0 &&
                                   sampled_inner_iters < num_hw_iters - 2);
                if (is_inner_iter_sampled) {
                    set_profiling_type_sampled(
                            sampled_inner_iters + 2, num_hw_iters);
                }
                // Used for the pipelined activation mechanism. The current
                // iteration will do activation function on the results produced
                // by previous iteration.
                int prev_iter_num_kerns = 0;
                int prev_kern_start = 0;
                for (int iter = 0; iter < num_hw_iters; iter++) {
                    bool is_last_iter = (iter == num_hw_iters - 1);
                    smv_convolution_options options;
                    options.img = img;
                    // This kern_start is with respect to the current set of
                    // output
                    // fmaps in the tile.
                    options.kern_start = iter * NUM_PE_INSTS;
                    int num_kerns_this_iter =
                            min2(output_tile->num_ofmaps - options.kern_start,
                                 (int)NUM_PE_INSTS);
                    options.kern_end = options.kern_start + num_kerns_this_iter;
                    // This is required to DMA the correct number of weights and
                    // outputs back from the accelerator at the beginning and
                    // end.
                    options.total_tile_ofmaps = output_tile->num_ofmaps;
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
                        INVOKE_KERNEL_PROF(g_smv->kConvolutionHw,
                                           lnum,
                                           smv_convolution_layer_hw,
                                           // DMA
                                           (packed_fp16*)activations_loc,
                                           (packed_fp16*)weights_loc,
                                           temp_result_buf,
                                           // Cache
                                           (packed_fp16*)activations_loc,
                                           (packed_fp16*)weights_loc,
                                           temp_result_buf,
                                           // ACP
                                           (packed_fp16*)activations_loc,
                                           (packed_fp16*)weights_loc,
                                           temp_result_buf,
                                           g_smv->umem,
                                           g_smv->spad0,
                                           g_smv->spad1,
                                           partial_layer,
                                           &access_cfg,
                                           &options);
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
                        begin_profiling(
                                "smv_convolution_layer_hw_activation_func",
                                lnum);
                        INVOKE_KERNEL_NOBLOCK(g_smv->kConvolutionHw,
                                              &finish_flag,
                                              smv_convolution_layer_hw,
                                              // DMA
                                              (packed_fp16*)activations_loc,
                                              (packed_fp16*)weights_loc,
                                              temp_result_buf,
                                              // Cache
                                              (packed_fp16*)activations_loc,
                                              (packed_fp16*)weights_loc,
                                              temp_result_buf,
                                              // ACP
                                              (packed_fp16*)activations_loc,
                                              (packed_fp16*)weights_loc,
                                              temp_result_buf,
                                              g_smv->umem,
                                              g_smv->spad0,
                                              g_smv->spad1,
                                              partial_layer,
                                              &access_cfg,
                                              &options);
                        if (iter != 0 && !do_hw_activation) {
                            begin_profiling(
                                    ACTIVATION_TYPE_STR(curr_layer.activation),
                                    lnum);
                            dims_t last_iter_dims = (dims_t){
                                partial_layer.outputs.rows,
                                partial_layer.outputs.cols,
                                prev_iter_num_kerns,
                                calc_padding(prev_iter_num_kerns, DATA_ALIGNMENT),
                            };
                            activation_fun_simd128(
                                    temp_result_buf,
                                    1,
                                    &last_iter_dims,
                                    curr_layer.activation,
                                    temp_result_buf);
                            end_profiling();
                        }
                        prev_iter_num_kerns = num_kerns_this_iter;
                        prev_kern_start = kern_start + options.kern_start;
                    }

                    if (iter != 0 && !is_last_iter) {
                        inner_iters_executed++;
                    }
                }
                // We need to do activation function for the last iteration of
                // the tile.
                if (use_pipelined_activation && !do_hw_activation) {
                    #ifdef GEM5_HARNESS
                    while (finish_flag == NOT_COMPLETED)
                        ;
                    #endif
                    end_profiling();
                    begin_profiling(
                            ACTIVATION_TYPE_STR(curr_layer.activation), lnum);
                    dims_t last_iter_dims = (dims_t){
                        partial_layer.outputs.rows,
                        partial_layer.outputs.cols,
                        prev_iter_num_kerns,
                        calc_padding(prev_iter_num_kerns, DATA_ALIGNMENT),
                    };
                    activation_fun_simd128(temp_result_buf,
                                           1,
                                           &last_iter_dims,
                                           curr_layer.activation,
                                           temp_result_buf);
                    end_profiling();
                }

                // Reorgnize the temporary results into the host result buffer.
                for (int k = 0; k < output_tile->num_ofmaps; k++) {
                    memcpy(&_result[img][k + kern_start][result_row_start][0],
                           temp_result_buf + (partial_result_2d_size * k) / 2,
                           partial_result_2d_size * sizeof(float16));
                }
                free(temp_result_buf);

                kern_start += output_tile->num_ofmaps;
                if (!is_last_output_tile) {
                    output_tiles_executed++;
                }
                end_profiling();
            }
            INFO_MSG("Finished an input tile.\n");
            kern_start = 0;
            input_row_start += (partial_layer.inputs.rows - halo_rows);
            result_row_start += partial_layer.outputs.rows;
        }
    }
    free_data_list(nhwc_activations);
    free_conv_tiling_cfg(&tiling);
}
