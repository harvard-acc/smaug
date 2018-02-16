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
                 "  Size: %d, %d, %d, %d, %d\n",
                 i,
                 cfg->tiles[i].input_dims[0],
                 cfg->tiles[i].input_dims[1],
                 cfg->tiles[i].input_dims[2],
                 cfg->tiles[i].input_dims[3],
                 cfg->tiles[i].input_dims[4]);
    }
}

static void smv_convolution_layer_hw_impl(float* dma_activations,
                                          float* dma_weights,
                                          float* dma_results,
                                          float* local_activations,
                                          float* local_weights,
                                          float* local_results,
                                          layer_t curr_layer,
                                          smv_convolution_options* options) {
    const int input_height = curr_layer.inputs.height;
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = curr_layer.inputs.align_pad;
    const int k_width = curr_layer.weights.cols;
    const int k_height = curr_layer.weights.height;
    const int k_pad = curr_layer.weights.align_pad;
    ARRAY_4D(float, _a, dma_activations, input_height + input_pad, input_rows,
             input_cols);
    ARRAY_4D(float, _kernels, dma_weights, k_width, k_width, k_height + k_pad);
    // DMA all the weights that we can fit in the current tile (which is
    // specified by this tile's outputs.height).
    // We should only DMA part of the weights.
    int single_weights_elems =
            curr_layer.weights.rows * curr_layer.weights.cols *
            (curr_layer.weights.height + curr_layer.weights.align_pad);
    int num_weights = options->total_tile_ofmaps * single_weights_elems;
    if (curr_layer.weights_req == IO_DMA) {
        setReadyBits(local_weights, num_weights * sizeof(float), 0);
        dmaLoad(local_weights, &_kernels[options->kern_start][0][0][0],
                num_weights * sizeof(float));
    }
    if (curr_layer.input_req == IO_DMA) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                curr_layer.inputs.rows * curr_layer.inputs.cols *
                (curr_layer.inputs.height + curr_layer.inputs.align_pad);
        setReadyBits(local_activations, num_input_pixels * sizeof(float), 0);
        dmaLoad(local_activations, &_a[options->img][0][0][0],
                num_input_pixels * sizeof(float));
    }

    // We will invoke the hardware multiple times, but we don't DMA every time.
    // So, we need to read weights from and write outputs to the kern_start'th
    // output channel.
    convolution3d_smv(local_activations, local_weights, curr_layer,
                      options->kern_start, local_results);

    // Run the activation function in-place if applicable.
    int ofmap_2d_elems =
            curr_layer.outputs.rows *
            (curr_layer.outputs.cols + curr_layer.outputs.align_pad);
    int num_output_pixels = options->total_tile_ofmaps * ofmap_2d_elems;
    smv_activation_fun(local_results, NUM_TEST_CASES, num_output_pixels,
                       curr_layer.activation);
    if (curr_layer.output_req == IO_DMA) {
        dmaStore(
                dma_results, local_results, num_output_pixels * sizeof(float));
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
//=--------- Convenience macros for invoking the HW impl ---------------=//
//
// Because the convolutional block cannot mix the use of scratchpads and the
// umem, we don't need the macros to help us select which spad to use, which
// reduces the number of macros greatly.

// No DMA or scratchpads are used at all.
#define CONV3D_NO_DMA_IMPL(INPUT, WGT, LR)                                     \
    do {                                                                       \
        PRINT_MSG(#INPUT "-" #WGT "-" #LR "\n");                               \
        smv_convolution_layer_hw_impl(NULL, NULL, NULL, INPUT##_activations,   \
                                      WGT##_weights, LR##_results, curr_layer, \
                                      options);                                \
    } while (0)

// Inputs can come from anywhere (dma, cache, or acp), and outputs can go
// anywhere.
#define CONV3D_WITH_DMA_IMPL(HA, HW, HR, LA, LW, LR)                           \
    do {                                                                       \
        PRINT_MSG(#HA "-" #HW "-" #HR "-" #LA "-" #LW "-" #LR "\n");           \
        smv_convolution_layer_hw_impl(HA##_activations, HW##_weights,          \
                                      HR##_results, LA, LW, LR, curr_layer,    \
                                      options);                                \
    } while (0)

    // These selections use the same mechanism all across.
    if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(dma, dma, dma, umem, spad0, spad1);
    } else if (DISPATCH_3(access_config, _ACP, _ACP, _ACP)) {
        CONV3D_NO_DMA_IMPL(acp, acp, acp);
    } else if (DISPATCH_3(access_config, _Cache, _Cache, _Cache)) {
        CONV3D_NO_DMA_IMPL(cache, cache, cache);
    }
    // These selections only use _ACP or _Cache for the results.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _ACP)) {
        CONV3D_WITH_DMA_IMPL(dma, dma, acp, umem, spad0, acp_results);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _Cache)) {
        CONV3D_WITH_DMA_IMPL(dma, dma, cache, umem, spad0, cache_results);
    }
    // These selections use DMA/None for the inputs.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _ACP, _ACP)) {
        CONV3D_WITH_DMA_IMPL(dma, acp, acp, umem, acp_weights, acp_results);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _Cache, _Cache)) {
        CONV3D_WITH_DMA_IMPL(
                dma, cache, cache, umem, cache_weights, cache_results);
    }
    // These selections use DMA/None for the inputs/outputs.
    //
    // NOTE: This scenario is currently not possible to specify via the model
    // configuration file.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _ACP, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(dma, acp, dma, umem, acp_weights, spad1);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _Cache, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(dma, cache, dma, umem, cache_weights, spad1);
    }
    // These selections use DMA/None for the outputs.
    else if (DISPATCH_3(access_config, _ACP, _ACP, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(
                acp, acp, dma, acp_activations, acp_weights, spad1);
    } else if (DISPATCH_3(access_config, _Cache, _Cache, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(
                cache, cache, dma, cache_activations, cache_weights, spad1);
    }
    // These selections only use DMA for the weights.
    else if (DISPATCH_3(access_config, _ACP, _DmaOrLocal, _ACP)) {
        CONV3D_WITH_DMA_IMPL(
                acp, dma, acp, acp_activations, spad0, acp_results);
    } else if (DISPATCH_3(access_config, _Cache, _DmaOrLocal, _Cache)) {
        CONV3D_WITH_DMA_IMPL(
                cache, dma, cache, cache_activations, spad0, cache_results);
    }
    // These selections use ACP/Cache for the inputs only.
    // This is usually used if the weights either needed DMA or are already in
    // the scratchpads.
    else if (DISPATCH_3(access_config, _ACP, _DmaOrLocal, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(
                acp, dma, dma, acp_activations, spad0, spad1);
    } else if (DISPATCH_3(access_config, _Cache, _DmaOrLocal, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(
                cache, dma, dma, cache_activations, spad0, spad1);
    }
    // Otherwise, give up.
    else {
        assert(false &&
               "This is an unsupported combination of access mechanisms!");
    }

#undef CONV3D_WITH_DMA_IMPL
#undef CONV3D_NO_DMA_IMPL
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

    if (total_input_bytes > SMV_UMEM_SIZE) {
        printf("A single input image exceeds the capacity of the UMEM, which "
               "is not supported!\n");
        assert(false);
    }

    const int output_2d_size =
            curr_layer->outputs.rows *
            (curr_layer->outputs.cols + curr_layer->outputs.align_pad) *
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
            get_nhwc_dims_size(&curr_layer->weights) * sizeof(float);
    const int max_kernels_per_iter = SMV_SPAD_SIZE / single_kernel_size;
    const int max_ofmaps_per_iter = SMV_SPAD_SIZE / output_2d_size;
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
    float* nhwc_activations = NULL;
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            host_activations, NUM_TEST_CASES, curr_layer.inputs, DATA_ALIGNMENT,
            &nhwc_activations);
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
            dims_t weights_nhwc = convert_nchw_to_nhwc(
                    &_kernels[kern_start][0][0][0], tile->num_ofmaps,
                    curr_layer.weights, DATA_ALIGNMENT, &nhwc_weights);
            MAP_ARRAY_TO_ACCEL(
                    g_smv->kConvolutionHw,
                    get_host_weights_var_name(curr_layer.weights_req),
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
                               sampled_inner_iters < tile->num_ofmaps - 2);
            if (is_sampled)
                set_profiling_type_sampled(
                        sampled_inner_iters + 2, tile->num_ofmaps);
            for (int iter = 0; iter < num_hw_iters; iter++) {
                bool is_last_iter = (iter == num_hw_iters - 1);
                int num_kerns_this_iter =
                        min2(tile->num_ofmaps - kern_start, NUM_PE_INSTS);
                smv_convolution_options options;
                options.img = img;
                // This kern_start is with respect to the current set of output
                // fmaps in the tile.
                options.kern_start = iter * NUM_PE_INSTS;
                options.kern_end = options.kern_start + num_kerns_this_iter;
                // This is required to DMA the correct number of weights and
                // outputs back from the accelerator at the beginning and end.
                options.total_tile_ofmaps = tile->num_ofmaps;

                if (iter != 0 && !is_last_iter &&
                    inner_iters_executed >= sampled_inner_iters) {
                    // Skip this iteration and zero out the result.
                    memset(&_result[img][options.kern_start][0][0], 0,
                           result_2d_size * num_kerns_this_iter);
                    continue;
                }

                // Only DMA inputs on the first iteration.
                if (iter > 0 && partial_layer.input_req == IO_DMA)
                    partial_layer.input_req = IO_NONE;
                // Only DMA weights on the first iteration.
                if (iter > 0 && partial_layer.weights_req == IO_DMA)
                    partial_layer.weights_req = IO_NONE;
                // Only DMA results back on the last.
                if (curr_layer.output_req == IO_DMA) {
                    if (!is_last_iter)
                        partial_layer.output_req = IO_NONE;
                    else
                        partial_layer.output_req = IO_DMA;
                }
                // Only run the activation function on the last iteration.
                partial_layer.activation = (is_last_iter && do_hw_activation)
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
