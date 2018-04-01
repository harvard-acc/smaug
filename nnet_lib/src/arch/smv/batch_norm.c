#include "arch/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _smv_batch_norm_options {
    smv_sram inputs_loc;
    smv_sram weights_loc;
    smv_sram outputs_loc;
    bool use_pipelined_dma;
    int weight_col_start;
} smv_batch_norm_options;

typedef struct _smv_batch_norm_output_tile {
    int dims[5];
    int pad;
    int num;
    int input_tile_chan_start;
} smv_batch_norm_output_tile;

typedef struct _smv_batch_norm_input_tile {
    int dims[5];
    int pad;
    int num;
    int input_chan_start;
    smv_batch_norm_output_tile* output_tiles;
    int num_output_tiles;
} smv_batch_norm_input_tile;

typedef struct _smv_batch_norm_tiling_cfg {
    smv_batch_norm_input_tile* input_tiles;
    int num_input_tiles;
} smv_batch_norm_tiling_cfg;

static smv_batch_norm_output_tile* init_smv_batch_norm_output_tiles(int num_tiles) {
    smv_batch_norm_output_tile* tiles = (smv_batch_norm_output_tile*)malloc(
            sizeof(smv_batch_norm_output_tile) * num_tiles);
    for (int i = 0; i < num_tiles; i++)
        tiles[i].num = i;
    return tiles;
}

static smv_batch_norm_tiling_cfg* init_smv_batch_norm_tiling_cfg(
        int num_tiles) {
    smv_batch_norm_tiling_cfg* cfg = (smv_batch_norm_tiling_cfg*)malloc(
            sizeof(smv_batch_norm_tiling_cfg));
    cfg->num_input_tiles = num_tiles;
    cfg->input_tiles = (smv_batch_norm_input_tile*)malloc(
            sizeof(smv_batch_norm_input_tile) * num_tiles);
    for (int i = 0; i < num_tiles; i++)
        cfg->input_tiles[i].num = i;
    return cfg;
}

static void print_smv_batch_norm_tiling_cfg(smv_batch_norm_tiling_cfg* cfg,
                                            int lnum) {
    INFO_MSG("\nTiling info for layer %d\n", lnum);
    INFO_MSG("\nNumber of input tiles: %d\n", cfg->num_input_tiles);
    for (int i = 0; i < cfg->num_input_tiles; i++) {
        smv_batch_norm_input_tile* input_tile =
                &cfg->input_tiles[i];
        INFO_MSG("Input tile %d\n"
                 "  IFMap size: %d, %d, %d, %d, %d\n"
                 "  input pad: %d\n"
                 "  Output tiles: %d\n",
                 i,
                 input_tile->dims[0],
                 input_tile->dims[1],
                 input_tile->dims[2],
                 input_tile->dims[3],
                 input_tile->dims[4],
                 input_tile->pad,
                 input_tile->num_output_tiles);
        for (int j = 0; j < input_tile->num_output_tiles; j++) {
            smv_batch_norm_output_tile* output_tile =
                    &input_tile->output_tiles[j];
            INFO_MSG("  + Output tile %d:\n"
                     "      OFMap size: %d, %d, %d, %d, %d\n"
                     "      output pad %d\n",
                     j,
                     output_tile->dims[0],
                     output_tile->dims[1],
                     output_tile->dims[2],
                     output_tile->dims[3],
                     output_tile->dims[4],
                     output_tile->pad);
        }
    }
}

static layer_t create_partial_layer_from_tile(
        layer_t* full_layer,
        smv_batch_norm_input_tile* input_tile,
        smv_batch_norm_output_tile* output_tile) {
    layer_t partial_layer = *full_layer;
    partial_layer.inputs.height = input_tile->dims[2];
    partial_layer.inputs.rows = input_tile->dims[1];
    partial_layer.inputs.cols = input_tile->dims[0];
    partial_layer.inputs.align_pad = input_tile->pad;
    partial_layer.outputs.height = output_tile->dims[2];
    partial_layer.outputs.rows = output_tile->dims[1];
    partial_layer.outputs.cols = output_tile->dims[0];
    partial_layer.outputs.align_pad = output_tile->pad;
    return partial_layer;
}

static smv_batch_norm_tiling_cfg* smv_batch_norm_tile_work(layer_t* curr_layer,
                                                           smv_global* g_smv) {
    int input_2d_size =
            curr_layer->inputs.rows *
            (curr_layer->inputs.cols + curr_layer->inputs.align_pad);
    // NOTE: since we only implement one-level tiling for batch norm, we use
    // kSpadSize here instead of kUmemSize.
    // This has to be a multiple of VECTOR_SIZE or it will fail the assert check
    // later on.
    // TODO: make the input to batch norm in NWHC format.
    int max_chans_per_input_tile =
            (g_smv->kSpadSize / (input_2d_size * sizeof(float)) / VECTOR_SIZE) *
            VECTOR_SIZE;
    int max_chans_per_output_tile =
            (g_smv->kSpadSize / (input_2d_size * sizeof(float)) / VECTOR_SIZE) *
            VECTOR_SIZE;
    int num_input_tiles =
            ceil(((float)curr_layer->inputs.height) / max_chans_per_input_tile);

    smv_batch_norm_tiling_cfg* cfg =
            init_smv_batch_norm_tiling_cfg(num_input_tiles);
    int num_input_chans_remaining = curr_layer->inputs.height;
    for (int i = 0; i < num_input_tiles; i++) {
        int input_chans_this_iter =
                min2(max_chans_per_input_tile, num_input_chans_remaining);
        cfg->input_tiles[i].dims[0] = curr_layer->inputs.cols;
        cfg->input_tiles[i].dims[1] = curr_layer->inputs.rows;
        cfg->input_tiles[i].dims[2] = input_chans_this_iter;
        cfg->input_tiles[i].dims[3] = NUM_TEST_CASES;
        cfg->input_tiles[i].dims[4] = 1;
        cfg->input_tiles[i].pad = curr_layer->inputs.align_pad;
        cfg->input_tiles[i].input_chan_start = i * max_chans_per_input_tile;

        int num_output_tiles = ceil(((float)cfg->input_tiles[i].dims[2]) /
                                    max_chans_per_output_tile);
        cfg->input_tiles[i].num_output_tiles = num_output_tiles;
        cfg->input_tiles[i].output_tiles =
                init_smv_batch_norm_output_tiles(num_output_tiles);
        int num_output_chans_remaining = cfg->input_tiles[i].dims[2];
        for (int j = 0; j < num_output_tiles; j++) {
            int output_chans_this_iter = min2(
                    max_chans_per_output_tile, num_output_chans_remaining);
            cfg->input_tiles[i].output_tiles[j].dims[0] =
                    curr_layer->outputs.cols;
            cfg->input_tiles[i].output_tiles[j].dims[1] =
                    curr_layer->outputs.rows;
            cfg->input_tiles[i].output_tiles[j].dims[2] =
                    output_chans_this_iter;
            cfg->input_tiles[i].output_tiles[j].dims[3] = NUM_TEST_CASES;
            cfg->input_tiles[i].output_tiles[j].dims[4] = 1;
            cfg->input_tiles[i].output_tiles[j].pad =
                    curr_layer->outputs.align_pad;
            cfg->input_tiles[i].output_tiles[j].input_tile_chan_start =
                    j * max_chans_per_output_tile;
            num_output_chans_remaining -= output_chans_this_iter;
        }
        num_input_chans_remaining -= input_chans_this_iter;
    }
    return cfg;
}

static void smv_batch_norm_layer_hw_impl(packed_fp16* host_activations,
                                         packed_fp16* host_weights,
                                         packed_fp16* host_results,
                                         float* local_activations,
                                         float* local_weights,
                                         float* local_results,
                                         layer_t* curr_layer,
                                         smv_batch_norm_options* options) {
    // DMA in the weights (to UMEM).
    int weights_size = get_num_weights_layer(curr_layer, 0);
    setReadyBits(local_weights, weights_size, 0);
    dma_load_and_unpack_fp16(local_weights, host_weights, weights_size, 0, 0);

    // Load in the inputs.
    int input_size = get_input_activations_size(curr_layer);
    if (curr_layer->input_req == IO_DMA) {
        setReadyBits(local_activations, input_size, 0);
        dma_load_and_unpack_fp16(
                local_activations, host_activations, input_size, 0, 0);
    } else if (curr_layer->input_req == IO_ACP) {
        acp_load_and_unpack_fp16(
                local_activations, host_activations, input_size, 0, 0);
    }

    // The main kernel
#ifdef ENABLE_SIMD_IMPL
    batch_norm_simd_fxp(local_activations,
                        local_weights,
                        curr_layer,
                        NUM_TEST_CASES,
                        local_results,
                        options->weight_col_start);
#else
    activation_type activation = curr_layer->activation;
    batch_norm_fxp(local_activations, local_weights, curr_layer, NUM_TEST_CASES,
                   local_results);
    activation_fun(local_results, NUM_TEST_CASES, input_size,
                   curr_layer.outputs.align_pad, activation);
#endif

    // DMA out the result (from SPAD1)
    int output_size = get_output_activations_size(curr_layer);
    if (curr_layer->output_req == IO_DMA) {
        dma_pack_and_store_fp16(host_results, local_results, output_size, 0, 0);
    } else if (curr_layer->output_req == IO_ACP) {
        acp_pack_and_store_fp16(host_results, local_results, output_size, 0, 0);
    }
}

static void smv_batch_norm_layer_hw(packed_fp16* dma_activations,
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
                                    layer_t* curr_layer,
                                    smv_batch_norm_options* options) {
    bool use_acp_results = (curr_layer->output_req == IO_ACP ||
                            curr_layer->output_req == IO_CACHE);
    bool use_acp_inputs = (curr_layer->input_req == IO_ACP ||
                           curr_layer->input_req == IO_CACHE);
    bool use_acp_weights = (curr_layer->weights_req == IO_ACP);

    // Check that all the data organizations are supported.
    if (options->weights_loc == SMV_UMEM) {
        ASSERT((options->inputs_loc == SMV_SPAD0 &&
                options->outputs_loc == SMV_SPAD1) &&
               "If BN weights are in UMEM, then inputs must be in "
               "SPAD0 and outputs in SPAD1!");
    } else if (options->inputs_loc == SMV_UMEM) {
        ASSERT((options->weights_loc == SMV_SPAD0 &&
                options->outputs_loc == SMV_SPAD1) &&
               "If BN inputs are in UMEM, then weights must be in "
               "SPAD0 and outputs in SPAD1!");
    } else {
        ASSERT(false && "This is an unsupported set of data locations.");
    }

    if (use_acp_results) {
        curr_layer->output_req = IO_ACP;
        if (use_acp_inputs) {
            curr_layer->input_req = IO_ACP;
            if (options->weights_loc == SMV_UMEM) {
                if (use_acp_weights) {
                    smv_batch_norm_layer_hw_impl(acp_activations, acp_weights,
                                                 acp_results, spad0, umem,
                                                 spad1, curr_layer, options);
                } else {
                    smv_batch_norm_layer_hw_impl(acp_activations, dma_weights,
                                                 acp_results, spad0, umem,
                                                 spad1, curr_layer, options);
                }
            } else if (options->inputs_loc == SMV_UMEM) {
                if (use_acp_weights) {
                    smv_batch_norm_layer_hw_impl(acp_activations, acp_weights,
                                                 acp_results, umem, spad0,
                                                 spad1, curr_layer, options);
                } else {
                    smv_batch_norm_layer_hw_impl(acp_activations, dma_weights,
                                                 acp_results, umem, spad0,
                                                 spad1, curr_layer, options);
                }
            }
        } else {
            if (curr_layer->input_req != IO_NONE)
                curr_layer->input_req = IO_DMA;
            if (options->weights_loc == SMV_UMEM) {
                if (use_acp_weights) {
                    smv_batch_norm_layer_hw_impl(dma_activations, acp_weights,
                                                 acp_results, spad0, umem,
                                                 spad1, curr_layer, options);
                } else {
                    smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                                 acp_results, spad0, umem,
                                                 spad1, curr_layer, options);
                }
            } else if (options->inputs_loc == SMV_UMEM) {
                if (use_acp_weights) {
                    smv_batch_norm_layer_hw_impl(dma_activations, acp_weights,
                                                 acp_results, umem, spad0,
                                                 spad1, curr_layer, options);
                } else {
                    smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                                 acp_results, umem, spad0,
                                                 spad1, curr_layer, options);
                }
            }
        }
    } else {
        if (curr_layer->input_req != IO_NONE)
            curr_layer->input_req = IO_DMA;
        if (options->weights_loc == SMV_UMEM) {
            if (use_acp_weights) {
                smv_batch_norm_layer_hw_impl(dma_activations, acp_weights,
                                             dma_results, spad0, umem, spad1,
                                             curr_layer, options);
            } else {
                smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                             dma_results, spad0, umem, spad1,
                                             curr_layer, options);
            }
        } else if (options->inputs_loc == SMV_UMEM) {
            if (use_acp_weights) {
                smv_batch_norm_layer_hw_impl(dma_activations, acp_weights,
                                             dma_results, umem, spad0, spad1,
                                             curr_layer, options);
            } else {
                smv_batch_norm_layer_hw_impl(dma_activations, dma_weights,
                                             dma_results, umem, spad0, spad1,
                                             curr_layer, options);
            }
        }
    }
}

void smv_batch_norm_layer_impl(data_list* activations,
                               layer_t* layers,
                               int lnum,
                               data_list* results,
                               smv_global* g_smv,
                               device_t* device) {
    layer_t curr_layer = layers[lnum];
    data_list* weights = curr_layer.host_weights;

    if (device->use_hw_batch_norm) {
        require_data_type(activations, 0, UncompressedHalfPrecision);
        require_data_type(weights, 0, UncompressedHalfPrecision);
        const int input_height = curr_layer.inputs.height;
        const int input_rows = curr_layer.inputs.rows;
        const int input_cols = curr_layer.inputs.cols;
        const int input_pad = curr_layer.inputs.align_pad;
        const int output_height = curr_layer.outputs.height;
        const int output_rows = curr_layer.outputs.rows;
        const int output_cols = curr_layer.outputs.cols;
        const int output_pad = curr_layer.outputs.align_pad;
        fp16array_t* bn_weights = weights->data[0].dense_hp;
        smv_batch_norm_tiling_cfg* cfg =
                smv_batch_norm_tile_work(&curr_layer, g_smv);
        print_smv_batch_norm_tiling_cfg(cfg, lnum);
        size_t weights_size = bn_weights->size;
        size_t inputs_size = get_dims_size(&curr_layer.inputs) * NUM_TEST_CASES;
        size_t outputs_size = get_dims_size(&curr_layer.outputs) * NUM_TEST_CASES;
        // If weights are larger, put weights in the UMEM; otherwise, put them
        // in the spad.
        smv_sram inputs_loc, weights_loc, outputs_loc;
        if (weights_size > inputs_size) {
            weights_loc = SMV_UMEM;
            inputs_loc = SMV_SPAD0;
            outputs_loc = SMV_SPAD1;
        } else {
            weights_loc = SMV_SPAD0;
            inputs_loc = SMV_UMEM;
            outputs_loc = SMV_SPAD1;
        }
        assert(inputs_size == outputs_size);
        if (!device->use_hw_activation_func)
            curr_layer.activation = NO_ACTIVATION;

        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        if (curr_layer.input_req == IO_DMA) {
            flush_cache_range(activations, inputs_size * sizeof(float16));
        }
        if (curr_layer.weights_req == IO_DMA) {
            flush_cache_range(bn_weights->d, weights_size * sizeof(float16));
        }
        end_profiling();

        packed_fp16* act_buf = activations->data[0].dense_hp->d;
        packed_fp16* wgt_buf = weights->data[0].dense_hp->d;
        packed_fp16* out_buf = results->data[0].dense_hp->d;
        ARRAY_4D(float16, _activations, act_buf, input_height,
                 input_rows, input_cols + input_pad);
        ARRAY_4D(float16, _results, out_buf, output_height, output_rows,
                 output_cols + output_pad);
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_inputs_var_name(curr_layer.input_req),
                           act_buf,
                           inputs_size * sizeof(float16));
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_weights_var_name(curr_layer.weights_req),
                           wgt_buf, weights_size * sizeof(float16));
        MAP_ARRAY_TO_ACCEL(g_smv->kBatchNormHw,
                           get_host_results_var_name(curr_layer.output_req),
                           out_buf,
                           outputs_size * sizeof(float16));
        smv_batch_norm_options options;
        options.inputs_loc = inputs_loc;
        options.weights_loc = weights_loc;
        options.outputs_loc = outputs_loc;
        options.use_pipelined_dma = device->use_pipelined_dma;

        for (int img = 0; img < NUM_TEST_CASES; img++) {
            for (int it = 0; it < cfg->num_input_tiles; it++) {
                smv_batch_norm_input_tile* input_tile = &cfg->input_tiles[it];
                int input_chan_start = input_tile->input_chan_start;
                // NOTE: for now, the number of output tiles for an input tile
                // will always be 1.
                for (int ot = 0; ot < input_tile->num_output_tiles; ot++) {
                    smv_batch_norm_output_tile* output_tile =
                            &input_tile->output_tiles[ot];
                    layer_t partial_layer = create_partial_layer_from_tile(
                            &curr_layer, input_tile, output_tile);
                    int result_chan_start = input_chan_start +
                                            output_tile->input_tile_chan_start;
                    options.weight_col_start = input_chan_start;
                    assert(options.weight_col_start % VECTOR_SIZE == 0);
                    packed_fp16* act =
                            (packed_fp16*)&_activations[img][input_chan_start]
                                                       [0][0];
                    packed_fp16* res =
                            (packed_fp16*)&_results[img][result_chan_start][0]
                                                   [0];
                    INVOKE_KERNEL_PROF(
                        g_smv->kBatchNormHw, lnum,
                        smv_batch_norm_layer_hw,
                        act, wgt_buf, res, // DMA
                        act, wgt_buf, res, // Cache
                        act, wgt_buf, res, // ACP
                        g_smv->umem, g_smv->spad0, g_smv->spad1,
                        &partial_layer, &options);
                }
            }
        }
    } else {
        begin_profiling(__func__, lnum);
        // The reference implementation is faster than MKL since we can
        // precompute some of the weights. We have an optimized MKL version,
        // but it just calls this same function, so there's no point going
        // through that overhead.
        require_data_type(weights, 0, Uncompressed);
        float* bn_weights = weights->data[0].dense->d;
        farray_t* fp32_activations = NULL;
        farray_t* fp32_results = NULL;
        if (activations->type[0] == UncompressedHalfPrecision) {
            fp32_activations =
                    unpack_data_fp16x4(activations->data[0].dense_hp, NULL);
            fp32_results = init_farray(
                    NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
                    false);
        } else {
            fp32_activations = activations->data[0].dense;
            fp32_results = results->data[0].dense;
        }

        batch_norm_fxp(fp32_activations->d, bn_weights, &curr_layer,
                       NUM_TEST_CASES, fp32_results->d);
        if (device->use_hw_activation_func) {
            size_t input_size = get_dims_size(&curr_layer.inputs);
            activation_fun(fp32_results->d, NUM_TEST_CASES, input_size,
                           curr_layer.outputs.align_pad, curr_layer.activation);
        }

        if (activations->type[0] == UncompressedHalfPrecision) {
            // Ugly - need to free the farray_t* container without freeing the
            // buffer it wraps.
            packed_fp16* results_buf = results->data[0].dense_hp->d;
            free(results->data[0].dense_hp);
            results->data[0].dense_hp =
                    pack_data_fp16(fp32_results, results_buf);
            free_farray(fp32_activations);
            free_farray(fp32_results);
        }
        end_profiling();
    }
}
