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
                                     bool input_in_spad0,
                                     layer_t* all_layers,
                                     layer_t partial_layer,
                                     int layer_num,
                                     int img,
                                     int kern,
                                     int start_chan) {

    layer_t curr_layer = all_layers[layer_num];
    const int input_height = curr_layer.inputs.height;
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = partial_layer.inputs.align_pad;
    const int k_width = partial_layer.weights.cols;
    const int k_height = partial_layer.weights.height;
    const int k_pad = partial_layer.weights.align_pad;
    ARRAY_4D(float, _a, host_activations, input_height + input_pad, input_rows,
             input_cols);
    ARRAY_4D(float, _kernels, host_weights, k_width, k_width, k_height + k_pad);
    // DMA all the weights that we can fit in the current tile (which is
    // specified by this tile's outputs.height).
    size_t num_weights =
            partial_layer.outputs.height * partial_layer.weights.rows *
            (partial_layer.weights.height + partial_layer.weights.align_pad) *
            partial_layer.weights.cols;
    if (input_in_spad0) {
        setReadyBits(spad0, num_weights * sizeof(float), 0);
        dmaLoad(spad0, &_kernels[kern][start_chan][0][0],
                num_weights * sizeof(float));
    } else {
        setReadyBits(spad1, num_weights * sizeof(float), 0);
        dmaLoad(spad1, &_kernels[kern][start_chan][0][0],
                num_weights * sizeof(float));
    }
    if (partial_layer.input_req == IO_DMA) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                partial_layer.inputs.rows *
                (partial_layer.inputs.height + partial_layer.inputs.align_pad) *
                (partial_layer.inputs.cols);
        setReadyBits(umem, num_input_pixels * sizeof(float), 0);
        dmaLoad(umem, &_a[img][0][0][0], num_input_pixels * sizeof(float));
    }

    if (input_in_spad0)
        convolution3d_smv(umem, spad0, partial_layer, start_chan, spad1);
    else
        convolution3d_smv(umem, spad1, partial_layer, start_chan, spad0);

    if (partial_layer.output_req == IO_DMA) {
        size_t num_output_pixels =
                partial_layer.outputs.rows * partial_layer.outputs.height *
                (partial_layer.outputs.cols + partial_layer.outputs.align_pad);
        if (input_in_spad0)
            dmaStore(host_results, spad1, num_output_pixels * sizeof(float));
        else
            dmaStore(host_results, spad0, num_output_pixels * sizeof(float));
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

// Tile the convolution for a single set of output feature maps.
//
// This is only required if the input itself does not fit into the UMEM. We
// currently don't support this, so the tiling doesn't actually do anything,
// but at least the framework is there for when we do need it.
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

    init_conv_tiling_cfg(&cfg, 1);
    cfg.tiles[0].input_dims[0] = curr_layer->inputs.height;
    cfg.tiles[0].input_dims[1] = curr_layer->inputs.cols;
    cfg.tiles[0].input_dims[2] = curr_layer->inputs.rows;
    cfg.tiles[0].input_dims[3] = NUM_TEST_CASES;
    cfg.tiles[0].input_dims[4] = 1;
    cfg.tiles[0].input_pad =
            calc_padding(cfg.tiles[0].input_dims[0], DATA_ALIGNMENT);
    cfg.tiles[0].num_ofmaps = NUM_PE_INSTS;
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
    const int result_3d_size =
            result_rows * (result_cols + result_pad) * result_height;
    const int single_kernel_size =
            get_dims_size(&curr_layer.weights) * sizeof(float);
    float* nhwc_activations = NULL;
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            host_activations, NUM_TEST_CASES, curr_layer.inputs, DATA_ALIGNMENT,
            &nhwc_activations);
    ARRAY_4D(float, _result, host_result, result_height, result_rows,
             result_cols + result_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);

    conv_tiling_cfg tiling = convolution_divide_work(&curr_layer);
    print_conv_tiling_cfg(&tiling);

    size_t host_conv_buffer_size =
            result_3d_size * tiling.num_tiles * sizeof(float);
    float* host_conv_buffer = (float*)malloc_aligned(host_conv_buffer_size);
    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);

    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_activations", nhwc_activations,
                       get_dims_size(&curr_layer.inputs) * sizeof(float));

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
        for (int kern = 0; kern < num_kerns_to_simulate; kern += NUM_PE_INSTS) {
            int num_kerns = min2(num_kerns_to_simulate - kern, NUM_PE_INSTS);
            // Convert weights to NHWC.
            float* nhwc_weights = NULL;
            dims_t weights_nhwc = convert_nchw_to_nhwc(
                    &_kernels[kern][0][0][0], num_kerns, curr_layer.weights,
                    DATA_ALIGNMENT, &nhwc_weights);
            MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_weights", nhwc_weights,
                               num_kerns * single_kernel_size);
            int start_chan = 0;
            for (int t = 0; t < tiling.num_tiles; t++) {
                conv_tile tile = tiling.tiles[t];
                layer_t partial_layer =
                        create_partial_layer_from_tile(&curr_layer, &tile);
                partial_layer.input_req = (kern == 0) ? IO_DMA : IO_NONE;
                MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_results",
                                   host_conv_buffer, host_conv_buffer_size);
                INVOKE_KERNEL_PROF(kConvolutionHw, lnum,
                                   convolution_layer_smv_hw, nhwc_activations,
                                   nhwc_weights, host_conv_buffer, g_umem,
                                   g_spad0, g_spad1, true, layers,
                                   partial_layer, lnum, img, 0, start_chan);
            }
            free(nhwc_weights);
            // TODO: Can we get rid of this memcpy?
            memcpy(&_result[img][kern][0][0], host_conv_buffer,
                   result_3d_size * sizeof(float));
        }
        end_profiling();
    }
    free(nhwc_activations);
    free_conv_tiling_cfg(&tiling);
    free(host_conv_buffer);
}
