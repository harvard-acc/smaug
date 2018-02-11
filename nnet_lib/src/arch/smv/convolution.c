#include <string.h>

#include "arch/smiv/dispatch_utils.h"
#include "arch/smiv_common.h"
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

typedef struct _convolution_options {
    int img;
    int kern_start;
    int kern_end;
    int total_tile_ofmaps;
} convolution_options;

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
                 "  Size: %d, %d, %d, %d, %d\n",
                 i,
                 cfg->tiles[i].input_dims[0],
                 cfg->tiles[i].input_dims[1],
                 cfg->tiles[i].input_dims[2],
                 cfg->tiles[i].input_dims[3],
                 cfg->tiles[i].input_dims[4]);
    }
}

static void convolution_layer_smv_hw(float* host_activations,
                                     float* host_weights,
                                     float* host_results,
                                     float* umem,
                                     float* spad0,
                                     float* spad1,
                                     layer_t curr_layer,
                                     convolution_options* options) {
    const int input_height = curr_layer.inputs.height;
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = curr_layer.inputs.align_pad;
    const int k_width = curr_layer.weights.cols;
    const int k_height = curr_layer.weights.height;
    const int k_pad = curr_layer.weights.align_pad;
    ARRAY_4D(float, _a, host_activations, input_height + input_pad, input_rows,
             input_cols);
    ARRAY_4D(float, _kernels, host_weights, k_width, k_width, k_height + k_pad);
    // DMA all the weights that we can fit in the current tile (which is
    // specified by this tile's outputs.height).
    // We should only DMA part of the weights.
    int single_weights_elems =
            curr_layer.weights.rows * curr_layer.weights.cols *
            (curr_layer.weights.height + curr_layer.weights.align_pad);
    int num_weights = options->total_tile_ofmaps * single_weights_elems;
    if (curr_layer.weights_req == IO_DMA) {
        setReadyBits(spad0, num_weights * sizeof(float), 0);
        dmaLoad(spad0, &_kernels[options->kern_start][0][0][0],
                num_weights * sizeof(float));
    }
    if (curr_layer.input_req == IO_DMA) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                curr_layer.inputs.rows * curr_layer.inputs.cols *
                (curr_layer.inputs.height + curr_layer.inputs.align_pad);
        setReadyBits(umem, num_input_pixels * sizeof(float), 0);
        dmaLoad(umem, &_a[options->img][0][0][0],
                num_input_pixels * sizeof(float));
    }

    // We will invoke the hardware multiple times, but we don't DMA every time.
    // So, we need to read weights from and write outputs to the kern_start'th
    // output channel.
    convolution3d_smv(umem, spad0, curr_layer, options->kern_start, spad1);

    if (curr_layer.output_req == IO_DMA) {
        int ofmap_2d_elems =
                curr_layer.outputs.rows *
                (curr_layer.outputs.cols + curr_layer.outputs.align_pad);
        int num_output_pixels = options->total_tile_ofmaps * ofmap_2d_elems;
        dmaStore(host_results, spad1, num_output_pixels * sizeof(float));
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
    conv_tiling_cfg cfg;
    int total_input_bytes =
            (get_dims_size(&curr_layer->inputs) * sizeof(float)) /
            NUM_TEST_CASES;

    if (total_input_bytes > UMEM_SIZE) {
        printf("A single input image exceeds the capacity of the UMEM, which "
               "is not supported!\n");
        assert(false);
    }

    const int output_2d_size =
            curr_layer->outputs.rows *
            (curr_layer->outputs.cols + curr_layer->outputs.align_pad) *
            sizeof(float);
    if (output_2d_size > SPAD_SIZE) {
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
            get_nhwc_dims_size(&curr_layer->weights) * sizeof(float);
    const int max_kernels_per_iter = SPAD_SIZE / single_kernel_size;
    const int max_ofmaps_per_iter = SPAD_SIZE / output_2d_size;
    const int num_ofmaps_per_iter =
            min2(max_kernels_per_iter, max_ofmaps_per_iter);
    const int num_iters =
            ceil(((float)curr_layer->outputs.height) / num_ofmaps_per_iter);
    init_conv_tiling_cfg(&cfg, num_iters);
    int remaining_ofmaps = curr_layer->outputs.height;
    for (int i = 0; i < num_iters; i++) {
        cfg.tiles[i].input_dims[0] = curr_layer->inputs.height;
        cfg.tiles[i].input_dims[1] = curr_layer->inputs.cols;
        cfg.tiles[i].input_dims[2] = curr_layer->inputs.rows;
        cfg.tiles[i].input_dims[3] = NUM_TEST_CASES;
        cfg.tiles[i].input_dims[4] = 1;
        cfg.tiles[i].input_pad =
                calc_padding(cfg.tiles[0].input_dims[0], DATA_ALIGNMENT);
        cfg.tiles[i].num_ofmaps = min2(remaining_ofmaps, num_ofmaps_per_iter);
        remaining_ofmaps -= num_ofmaps_per_iter;
    }
    return cfg;
}

void standard_convolution_layer_smv_impl(float* host_activations,
                                         float* host_weights,
                                         layer_t* layers,
                                         int lnum,
                                         float* host_result,
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
    const int single_kernel_size =
            get_dims_size(&curr_layer.weights) * sizeof(float);
    float* nhwc_activations = NULL;
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            host_activations, NUM_TEST_CASES, curr_layer.inputs, DATA_ALIGNMENT,
            &nhwc_activations);
    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_activations", nhwc_activations,
                       get_dims_size(&curr_layer.inputs) * sizeof(float));

    ARRAY_4D(float, _result, host_result, result_height, result_rows,
             result_cols + result_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);

    conv_tiling_cfg tiling = convolution_divide_work(&curr_layer);
    print_conv_tiling_cfg(&tiling);

    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    // Sampling: if set, only run up to the specified number of output
    // channels.  Set all remaining outputs to zero.
    const int sample_num_kerns = sampling_param->standard_conv_num_filters;
    const int num_kerns_to_simulate =
            sample_num_kerns == 0 ? num_kerns
                                  : min2(num_kerns, sample_num_kerns);
    bool is_sampled = num_kerns_to_simulate < num_kerns;
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        begin_profiling(__func__, lnum);
        if (is_sampled)
            set_profiling_type_sampled(num_kerns_to_simulate, num_kerns);

        int kern_start = 0;
        for (int t = 0; t < tiling.num_tiles; t++) {
            conv_tile* tile = &tiling.tiles[t];
            layer_t partial_layer =
                    create_partial_layer_from_tile(&curr_layer, tile);
            if (t != 0)
                partial_layer.input_req = IO_NONE;

            // Set up the results buffer and mappings.
            float* result_loc = &_result[img][kern_start][0][0];
            int result_size = result_2d_size * tile->num_ofmaps * sizeof(float);
            MAP_ARRAY_TO_ACCEL(
                    kConvolutionHw, "host_results", result_loc, result_size);

            // Convert weights to NHWC and set up mappings.
            float* nhwc_weights = NULL;
            dims_t weights_nhwc = convert_nchw_to_nhwc(
                    &_kernels[kern_start][0][0][0], tile->num_ofmaps,
                    curr_layer.weights, DATA_ALIGNMENT, &nhwc_weights);
            MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_weights", nhwc_weights,
                               num_kerns * single_kernel_size);

            int num_hw_iters = ceil((float)tile->num_ofmaps / NUM_PE_INSTS);
            for (int iter = 0; iter < num_hw_iters; iter++) {
                int num_kerns =
                        min2(num_kerns_to_simulate - kern_start, NUM_PE_INSTS);
                convolution_options options;
                options.img = img;
                // This kern_start is with respect to the current set of output
                // fmaps in the tile.
                options.kern_start = iter * NUM_PE_INSTS;
                options.kern_end = options.kern_start + num_kerns;
                // This is required to DMA the correct number of weights and
                // outputs back from the accelerator at the beginning and end.
                options.total_tile_ofmaps = tile->num_ofmaps;
                // Only DMA weights on the first iteration.
                if (iter > 0)
                    partial_layer.weights_req = IO_NONE;
                // Only DMA results back on the last.
                partial_layer.output_req = (iter < num_hw_iters - 1)
                                                   ? IO_NONE
                                                   : curr_layer.output_req;
                INVOKE_KERNEL_PROF(kConvolutionHw, lnum,
                                   convolution_layer_smv_hw, nhwc_activations,
                                   nhwc_weights, result_loc, g_umem, g_spad0,
                                   g_spad1, partial_layer, &options);
            }
            free(nhwc_weights);
            kern_start += tile->num_ofmaps;
        }
        end_profiling();
    }
    free(nhwc_activations);
    free_conv_tiling_cfg(&tiling);
}
