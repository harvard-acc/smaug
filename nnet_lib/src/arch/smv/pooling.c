#include <assert.h>

#include "arch/common.h"
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
    // When running an output tile, start reading input pixels from this
    // channel in the UMEM.
    int pool_start_chan;
    // If true, put the output pixels in spad0; otherwise, put it in spad1.
    bool output_in_spad0;
} smv_pooling_options;

// A pooling output tile expresses the largest chunk of work whose output
// pixels can fit in a single scratchpad.
typedef struct _pool_output_tile {
    int output_dims[5];
    // Padding to be added to the innermost dimension (height).
    int output_pad;
    // See smv_pooling_options->pool_start_chan.
    int start_chan;
    // The N'th output tile in this input tile.
    int num;
    // Sampling parameters.
    int sampling_upscale_factor;
    bool execute;
} pool_output_tile;

// A pooling input tile expresses the largest chunk of input pixels that can
// fit in the UMEM. Multiple output tiles will be required to fully compute the
// results.
typedef struct _pool_input_tile {
    int input_dims[5];
    int input_pad;
    // This input tile starts from this channel in the full input image.
    int start_chan;
    // The N'th input tile in the full input image.
    int num;
    pool_output_tile* output_tiles;
    int num_output_tiles;
} pool_input_tile;

typedef struct _pool_tiling_cfg {
    pool_input_tile* input_tiles;
    int num_input_tiles;
} pool_tiling_cfg;

static void smv_pooling_layer_hw_impl(packed_fp16* host_activations,
                                      packed_fp16* host_results,
                                      float* local_activations,
                                      float* local_results,
                                      layer_t curr_layer,
                                      smv_pooling_options* options) {
    int partial_input_2d_size = curr_layer.inputs.rows * curr_layer.inputs.cols;
    int partial_input_size =
            partial_input_2d_size *
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
    if (curr_layer.pool == MAX) {
        maxpooling_nhwc_smiv(local_activations, curr_layer,
                             options->pool_start_chan, local_results);
    } else {
        avgpooling_nhwc_smiv(local_activations, curr_layer,
                             options->pool_start_chan, local_results);
    }

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
            if (options->output_in_spad0) {
                smv_pooling_layer_hw_impl(acp_activations, acp_results, umem,
                                          spad0, curr_layer, options);
            } else {
                smv_pooling_layer_hw_impl(acp_activations, acp_results, umem,
                                          spad1, curr_layer, options);
            }
        } else {
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            if (options->output_in_spad0) {
                smv_pooling_layer_hw_impl(dma_activations, acp_results, umem,
                                          spad0, curr_layer, options);
            } else {
                smv_pooling_layer_hw_impl(dma_activations, acp_results, umem,
                                          spad1, curr_layer, options);
            }
        }
    } else {
        if (options->output_in_spad0) {
            smv_pooling_layer_hw_impl(dma_activations, dma_results, umem,
                                      spad0, curr_layer, options);
        } else {
            smv_pooling_layer_hw_impl(dma_activations, dma_results, umem,
                                      spad1, curr_layer, options);
        }
    }
}

pool_tiling_cfg* init_pool_tiling_cfg(int num_input_tiles) {
    pool_tiling_cfg* pool_cfg =
            (pool_tiling_cfg*)malloc(sizeof(pool_tiling_cfg));
    pool_cfg->input_tiles =
            (pool_input_tile*)malloc(sizeof(pool_input_tile) * num_input_tiles);
    pool_cfg->num_input_tiles = num_input_tiles;
    for (int i = 0; i < num_input_tiles; i++) {
        pool_cfg->input_tiles[i].num = i;
    }
    return pool_cfg;
}


pool_output_tile* init_pool_output_tiles(int num_output_tiles) {
    pool_output_tile* output_tiles = (pool_output_tile*)malloc(
            sizeof(pool_output_tile) * num_output_tiles);
    for (int i = 0; i < num_output_tiles; i++) {
        output_tiles[i].num = i;
        output_tiles[i].sampling_upscale_factor = 1;
        output_tiles[i].execute = true;
    }
    return output_tiles;
}

/* Divide up the input across input and output tiles.
 *
 * The pooling input is a blocked NHWC format. Each block is a H * W * B
 * volume, where B = blocking size (VECTOR_SIZE). This is the minimum amount of
 * work that must be done. Each input tile is composed of an integral number of
 * blocks. For each input tile, we divide that up into M output tiles, where M
 * is the largest output size that will fit into the scratchpad.
 */
static pool_tiling_cfg* smv_pooling_divide_work(layer_t* curr_layer,
                                                smv_global* g_smv) {

    size_t total_input_bytes =
            get_nhwc_dims_size(&curr_layer->inputs) * sizeof(float);
    bool need_input_tiling = (total_input_bytes > g_smv->kUmemSize);
    const int max_chans_per_block = VECTOR_SIZE;
    size_t output_block_size = curr_layer->outputs.rows *
                               curr_layer->outputs.cols * max_chans_per_block *
                               sizeof(float);
    if (output_block_size > g_smv->kSpadSize) {
        fprintf(stderr,
                "A single output tile doesn't fit on the scratchpad! We "
                "don't support this mode of tiling yet!\n");
        assert(false);
    }

    int max_blocks_per_output_tile = g_smv->kSpadSize / output_block_size;
    int max_chans_per_input_tile, num_input_tiles;
    if (!need_input_tiling) {
        // If the whole input can fit on the UMEM, initialize the variables with
        // single input tile setting.
        num_input_tiles = 1;
        max_chans_per_input_tile = curr_layer->inputs.height;
    } else {
        size_t input_block_size = curr_layer->inputs.rows *
                                  curr_layer->inputs.cols *
                                  max_chans_per_block * sizeof(float);
        int max_blocks_per_input_tile = g_smv->kUmemSize / input_block_size;
        if (input_block_size > g_smv->kUmemSize) {
            printf("A single tile of the input image exceeds the capacity of "
                   "the UMEM! This is not supported!\n");
            assert(false);
        }
        // Divide up the input over blocks.
        max_chans_per_input_tile =
                max_blocks_per_input_tile * max_chans_per_block;
        num_input_tiles = ceil(((float)(curr_layer->inputs.height)) /
                               (max_chans_per_input_tile));
    }

    // Create tiling configurations.
    pool_tiling_cfg* cfg = init_pool_tiling_cfg(num_input_tiles);
    int remaining_input_chans = curr_layer->inputs.height;
    for (int i = 0; i < cfg->num_input_tiles; i++) {
        pool_input_tile* input_tile = &cfg->input_tiles[i];
        int input_tile_chans =
                min2(remaining_input_chans, max_chans_per_input_tile);
        input_tile->input_dims[0] = input_tile_chans;
        input_tile->input_dims[1] = curr_layer->inputs.rows;
        input_tile->input_dims[2] = curr_layer->inputs.cols;
        input_tile->input_dims[3] = NUM_TEST_CASES;
        input_tile->input_dims[4] = 1;
        input_tile->input_pad = calc_padding(input_tile_chans, DATA_ALIGNMENT);
        input_tile->start_chan = i * max_chans_per_input_tile;
        int total_output_size = curr_layer->outputs.rows *
                                curr_layer->outputs.cols * input_tile_chans *
                                sizeof(float);
        int num_output_tiles =
                ceil(((float)total_output_size) / g_smv->kSpadSize);
        input_tile->num_output_tiles = num_output_tiles;
        input_tile->output_tiles = init_pool_output_tiles(num_output_tiles);
        int max_chans_per_output_tile =
                max_blocks_per_output_tile * max_chans_per_block;
        int remaining_chans = input_tile_chans;
        for (int j = 0; j < input_tile->num_output_tiles; j++) {
            pool_output_tile* output_tile = &input_tile->output_tiles[j];
            int num_output_channels =
                    min2(remaining_chans, max_chans_per_output_tile);
            output_tile->output_dims[0] = num_output_channels;
            output_tile->output_dims[1] = curr_layer->outputs.rows;
            output_tile->output_dims[2] = curr_layer->outputs.cols;
            output_tile->output_dims[3] = NUM_TEST_CASES;
            output_tile->output_dims[4] = 1;
            output_tile->output_pad =
                    calc_padding(num_output_channels, DATA_ALIGNMENT);
            output_tile->start_chan = j * max_chans_per_output_tile;
            remaining_chans -= max_chans_per_output_tile;
        }
        remaining_input_chans -= max_chans_per_input_tile;
    }
    return cfg;
}

void free_pool_tiling_cfg(pool_tiling_cfg* cfg) {
    for (int i = 0; i < cfg->num_input_tiles; i++) {
        free(cfg->input_tiles[i].output_tiles);
    }
    free(cfg->input_tiles);
    free(cfg);
}

void print_pool_tiling_cfg(pool_tiling_cfg* cfg, int lnum) {
    INFO_MSG("\nTiling info for layer %d\n", lnum);
    for (int i = 0; i < cfg->num_input_tiles; i++) {
        pool_input_tile* input_tile =
                &cfg->input_tiles[i];
        INFO_MSG("Input tile %d\n"
                 "  IFMap size: %d, %d, %d, %d, %d\n"
                 "  Input pad: %d\n"
                 "  Input start chan: %d\n"
                 "  Output tiles: %d\n",
                 i,
                 input_tile->input_dims[0],
                 input_tile->input_dims[1],
                 input_tile->input_dims[2],
                 input_tile->input_dims[3],
                 input_tile->input_dims[4],
                 input_tile->input_pad,
                 input_tile->start_chan,
                 input_tile->num_output_tiles);
        for (int j = 0; j < input_tile->num_output_tiles; j++) {
            pool_output_tile* output_tile =
                    &input_tile->output_tiles[j];
            INFO_MSG("  + Output tile %d:\n"
                     "      Execute: %s\n"
                     "      OFMap size: %d, %d, %d, %d, %d\n"
                     "      Output pad: %d\n"
                     "      Input start chan (relative to input tile): %d\n"
                     "      Each tile represents: %d output tiles\n",
                     j,
                     bool_to_yesno(output_tile->execute),
                     output_tile->output_dims[0],
                     output_tile->output_dims[1],
                     output_tile->output_dims[2],
                     output_tile->output_dims[3],
                     output_tile->output_dims[4],
                     output_tile->output_pad,
                     output_tile->start_chan,
                     output_tile->sampling_upscale_factor);
        }
    }
}

layer_t create_partial_pool_layer(layer_t* curr_layer,
                                  device_t* device,
                                  pool_input_tile* input_tile,
                                  pool_output_tile* output_tile) {
    layer_t partial_layer;
    partial_layer.type = curr_layer->type;
    partial_layer.num = curr_layer->num;
    partial_layer.activation = curr_layer->activation;
    partial_layer.pool = curr_layer->pool;
    partial_layer.input_preprocessing = curr_layer->input_preprocessing;

    partial_layer.inputs.height = input_tile->input_dims[0];
    partial_layer.inputs.rows = input_tile->input_dims[1];
    partial_layer.inputs.cols = input_tile->input_dims[2];
    partial_layer.inputs.align_pad = input_tile->input_pad;
    partial_layer.weights.rows = curr_layer->weights.rows;
    partial_layer.weights.cols = curr_layer->weights.cols;
    partial_layer.weights.height = curr_layer->weights.height;
    partial_layer.weights.align_pad = curr_layer->weights.align_pad;
    partial_layer.outputs.height = output_tile->output_dims[0];
    partial_layer.outputs.rows = output_tile->output_dims[1];
    partial_layer.outputs.cols = output_tile->output_dims[2];
    partial_layer.outputs.align_pad = output_tile->output_pad;
    partial_layer.biases.rows = 0;
    partial_layer.biases.cols = 0;
    partial_layer.biases.height = 0;
    partial_layer.biases.align_pad = 0;
    partial_layer.stride.rows = curr_layer->stride.rows;
    partial_layer.stride.cols = curr_layer->stride.cols;

    partial_layer.input_req =
            output_tile->num == 0 ? curr_layer->input_req : IO_NONE;
    partial_layer.output_req = curr_layer->output_req != IO_NONE
                                       ? curr_layer->output_req
                                       : device->cpu_default_offload;
    return partial_layer;
}

void smv_pooling_layer_impl(data_list* inputs,
                            layer_t* curr_layer,
                            smv_global* g_smv,
                            data_list* results,
                            device_t* device) {
    require_data_type(inputs, 0, UncompressedHalfPrecision);
    begin_profiling("smv_pooling_divide_work", curr_layer->num);
    pool_tiling_cfg* pool_cfg = smv_pooling_divide_work(curr_layer, g_smv);
    print_pool_tiling_cfg(pool_cfg, curr_layer->num);
    end_profiling();

    begin_profiling("convert_nchw_to_blocked_nhwc", curr_layer->num);
    data_list* nhwc_inputs = init_data_list(1);
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
        float16* current_inputs = (float16*)nhwc_inputs->data[0].dense_hp->d;
        float16* current_results = (float16*)nhwc_outputs->data[0].dense_hp->d;
        // Flush cache lines for inputs and outputs.
        begin_ignored_profiling(curr_layer->num);
        if (curr_layer->input_req == IO_DMA) {
            flush_cache_range(
                    current_inputs,
                    nhwc_inputs->data[0].dense_hp->size * sizeof(float16));
        }
        if (curr_layer->output_req == IO_DMA) {
            flush_cache_range(
                    current_results,
                    nhwc_outputs->data[0].dense_hp->size * sizeof(float16));
        }
        end_profiling();

        for (int input_tile_num = 0;
             input_tile_num < pool_cfg->num_input_tiles; input_tile_num++) {
            pool_input_tile* input_tile =
                    &pool_cfg->input_tiles[input_tile_num];
            int input_tile_size =
                    (input_tile->input_dims[0] + input_tile->input_pad) *
                    input_tile->input_dims[1] * input_tile->input_dims[2];
            MAP_ARRAY_TO_ACCEL(
                    g_smv->kPoolingHw,
                    get_host_inputs_var_name(curr_layer->input_req),
                    current_inputs,
                    input_tile_size * sizeof(float16));

            for (int output_tile_num = 0;
                 output_tile_num < input_tile->num_output_tiles;
                 output_tile_num++) {
                pool_output_tile* output_tile =
                        &input_tile->output_tiles[output_tile_num];
                layer_t partial_layer = create_partial_pool_layer(
                        curr_layer, device, input_tile, output_tile);
                int output_tile_size =
                        get_nhwc_dims_size(&partial_layer.outputs);
                MAP_ARRAY_TO_ACCEL(
                        g_smv->kPoolingHw,
                        get_host_results_var_name(partial_layer.output_req),
                        current_results,
                        output_tile_size * sizeof(float16));

                smv_pooling_options options;
                options.pool_start_chan = output_tile->start_chan;
                options.output_in_spad0 = true;
                INVOKE_KERNEL_PROF(g_smv->kPoolingHw,
                                   partial_layer.num,
                                   smv_pooling_layer_hw,
                                   (packed_fp16*)current_inputs,  // DMA
                                   (packed_fp16*)current_results,
                                   (packed_fp16*)current_inputs,  // Cache
                                   (packed_fp16*)current_results,
                                   (packed_fp16*)current_inputs,  // ACP
                                   (packed_fp16*)current_results,
                                   g_smv->umem,
                                   g_smv->spad0,
                                   g_smv->spad1,
                                   partial_layer,
                                   &options);
                current_results += output_tile_size;
            }
            current_inputs += input_tile_size;
        }
    }

    begin_profiling("convert_blocked_nhwc_to_nhwc", curr_layer->num);
    dims_t output_dims =
            nchw_to_nhwc_dims(&curr_layer->outputs, DATA_ALIGNMENT);
    convert_blocked_nhwc_to_nchw(nhwc_outputs, 0, NUM_TEST_CASES, VECTOR_SIZE,
                                 output_dims, DATA_ALIGNMENT, results);
    end_profiling();

    free_data_list(nhwc_inputs);
    free_data_list(nhwc_outputs);
    free_pool_tiling_cfg(pool_cfg);
}
