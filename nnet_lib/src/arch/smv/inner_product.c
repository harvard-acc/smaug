#include <assert.h>
#include <string.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/activation_functions_simd.h"
#include "core/smiv/params.h"
#include "core/smv/params.h"
#include "core/smv/smv.h"
#include "utility/compression.h"
#include "utility/utility.h"
#include "config.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef void (*tiled_inner_product_impl)(
        float*, float*, layer_t*, float*, device_t*, bool);

typedef struct _eltwise_options {
    bool do_bias;
    bool input_in_spad0;
} smv_eltwise_options;

typedef struct _inner_product_options {
    bool input_in_spad0;
    bool use_pipelined_dma;
    bool accumulate;
    io_req_t psums_req;
    int result_start;
    int dma_store_start;
} smv_inner_product_options;

typedef struct _inner_product_strip {
    int weights_dims[2];
    int weights_start_row;
    int num;
} inner_product_strip;

typedef struct _inner_product_tile {
    int num;
    int inputs_start_offset;
    inner_product_strip* strips;
    int num_strips;
} inner_product_tile;

typedef struct _inner_product_tiling_cfg {
    inner_product_tile* tiles;
    int num_tiles;
} inner_product_tiling_cfg;

typedef struct _smv_decompression_options {
    int tile_num;
    int current_row;
    bool input_in_spad0;
} smv_decompression_options;

void smv_eltwise_hw_impl(packed_fp16* host_activations,
                         packed_fp16* host_weights,
                         float* local_activations,
                         float* local_weights,
                         layer_t* curr_layer,
                         smv_eltwise_options* options) {
    if (curr_layer->weights_req != IO_NONE) {
        ASSERT(host_weights && "DMA weights pointer cannot be NULL!");
        int weights_size = get_num_weights_layer(curr_layer, 0);
        setReadyBits(local_weights, weights_size * sizeof(float), 0);
        dma_load_and_unpack_fp16(
                local_weights, host_weights, weights_size, 0, 0);
    }

    if (curr_layer->input_req != IO_NONE) {
        ASSERT(host_activations && "DMA inputs pointer cannot be NULL!");
        int activations_size = get_input_activations_size(curr_layer);
        if (curr_layer->input_req == IO_DMA) {
            setReadyBits(
                    local_activations, activations_size * sizeof(float), 0);
            dma_load_and_unpack_fp16(local_activations,
                                     host_activations,
                                     activations_size, 0, 0);
        } else {
            acp_load_and_unpack_fp16(local_activations, host_activations,
                                     activations_size, 0, 0);
        }
    }

    VEC_ARRAY_1D(v8fp_t, _local_activations, local_activations);
    if (options->do_bias || curr_layer->activation != NO_ACTIVATION) {
        int output_cols = curr_layer->outputs.cols;
        VEC_ARRAY_1D(v8fp_t, _weights, local_weights);
        const v8fp_t zero = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
        bias_and_act_func:
        for (int i = 0; i < FRAC_CEIL(output_cols, VECTOR_SIZE); i++) {
            v8fp_t psum = _local_activations[i] +
                          (options->do_bias ? _weights[i] : zero);
            _local_activations[i] =
                    activation_fun_simd_fxp(psum, curr_layer->activation);
        }
    }

    size_t result_size = get_output_activations_size(curr_layer);
    if (curr_layer->output_req == IO_ACP ||
        curr_layer->output_req == IO_CACHE) {
        acp_pack_and_store_fp16(
                host_activations, local_activations, result_size, 0, 0);
    } else if (curr_layer->output_req == IO_DMA) {
        ASSERT(host_activations && "DMA activations pointer cannot be NULL!");
        dma_pack_and_store_fp16(
                host_activations, local_activations, result_size, 0, 0);
    }
}

void smv_eltwise_hw(packed_fp16* dma_activations,
                    packed_fp16* dma_weights,
                    packed_fp16* cache_activations,
                    packed_fp16* cache_weights,
                    packed_fp16* acp_activations,
                    packed_fp16* acp_weights,
                    float* umem,
                    float* spad0,
                    float* spad1,
                    layer_t* curr_layer,
                    access_config* access_config,
                    smv_eltwise_options* options) {
    // We don't currently support using a local cache for inner products.  If
    // the IO requirement is IO_CACHE, it will be treated as IO_ACP.
    bool use_acp_results = (access_config->outputs == _ACP ||
                            access_config->outputs == _Cache);
    bool use_acp_inputs =
            (access_config->inputs == _ACP || access_config->inputs == _Cache);

    if (options->input_in_spad0) {
        if (use_acp_results) {
            if (use_acp_inputs) {
                smv_eltwise_hw_impl(acp_activations, dma_weights, spad0, umem,
                                    curr_layer, options);
            } else {
                // Not a common scenario but should be supported anyways.
                smv_eltwise_hw_impl(dma_activations, dma_weights, spad0, umem,
                                    curr_layer, options);
            }
        } else {
            if (use_acp_inputs) {
                // Common use case: Use ACP to load input, but leave data in
                // the spads.
                smv_eltwise_hw_impl(acp_activations, dma_weights, spad0, umem,
                                    curr_layer, options);
            } else {
                ASSERT((access_config->inputs == _DmaOrLocal &&
                        access_config->outputs == _DmaOrLocal) &&
                       "IO requirements are inconsistent with DMA fallback!");
                smv_eltwise_hw_impl(dma_activations, dma_weights, spad0, umem,
                                    curr_layer, options);
            }
        }
    } else {
        if (use_acp_results) {
            if (use_acp_inputs) {
                smv_eltwise_hw_impl(acp_activations, dma_weights, spad1, umem,
                                    curr_layer, options);
            } else {
                // Not a common scenario but should be supported anyways.
                smv_eltwise_hw_impl(dma_activations, dma_weights, spad1, umem,
                                    curr_layer, options);
            }
        } else {
            if (use_acp_inputs) {
                // Common use case: Use ACP to load input, but leave data in
                // the spads.
                smv_eltwise_hw_impl(acp_activations, dma_weights, spad1, umem,
                                    curr_layer, options);
            } else {
                ASSERT((access_config->inputs == _DmaOrLocal &&
                        access_config->outputs == _DmaOrLocal) &&
                       "IO requirements are inconsistent with DMA fallback!");
                smv_eltwise_hw_impl(dma_activations, dma_weights, spad1, umem,
                                    curr_layer, options);
            }
        }
    }
}

// Main implementation of inner product HW.
//
// This function handles DMA operations to load the local memory if required,
// runs the matrix multiply, and then handles the DMA transfer back to the host
// if required. In the context of this function, "local memory" is any type of
// memory that the accelerator can access directly (i.e. no DMA required), so
// it is either a scratchpad/umem or a private cache. "host memory" refers to
// either a DMA memory block or an ACP memory block -- for these, we will use
// either DMA or ACP to copy the data into a local scratchpad first before
// using the data.
//
// Arguments:
//   host_activations: The host address of the input activations.
//   host_weights: The host address of the input weights.
//   host_results: The host address of the input results.
//   local_activations: Pointer to inputs that the accelerator reads directly.
//   local_weights: Pointer to weights that the accelerator reads directly.
//   local_results: Pointer to results that the accelerator writes directly.
//   curr_layer: Description of this layer's shape and parameters.
//   options: Additional options for this execution of inner product.
void smv_inner_product_layer_hw_impl(packed_fp16* host_activations,
                                     packed_fp16* host_weights,
                                     packed_fp16* host_results,
                                     float* local_activations,
                                     float* local_weights,
                                     float* local_results,
                                     layer_t* curr_layer,
                                     smv_inner_product_options* options) {
    if (curr_layer->weights_req != IO_NONE) {
        ASSERT(host_weights && "DMA weights pointer cannot be NULL!");
        int weights_size = get_num_weights_layer(curr_layer, 0);
        setReadyBits(local_weights, weights_size, 0);
        dma_load_and_unpack_fp16(
                local_weights, host_weights, weights_size, 0, 0);
    }

    if (curr_layer->input_req == IO_DMA || curr_layer->input_req == IO_ACP ||
        curr_layer->input_req == IO_CACHE) {
        ASSERT(host_activations && "DMA inputs pointer cannot be NULL!");
        int activations_size = get_input_activations_size(curr_layer);
        if (curr_layer->input_req == IO_DMA) {
            setReadyBits(local_activations, activations_size, 0);
            dma_load_and_unpack_fp16(local_activations,
                                     host_activations,
                                     activations_size, 0, 0);
        } else {
            acp_load_and_unpack_fp16(local_activations, host_activations,
                                     activations_size, 0, 0);
        }
    }

    size_t result_size = get_output_activations_size(curr_layer);
    if (options->accumulate && options->psums_req != IO_NONE) {
        if (options->psums_req == IO_DMA) {
            dma_load_and_unpack_fp16(local_results, host_results, result_size,
                                     options->result_start,
                                     options->result_start);
        } else {
            acp_load_and_unpack_fp16(local_results, host_results, result_size,
                                     options->result_start,
                                     options->result_start);
        }
    }

    matrix_multiply_transpose_smv(
            local_activations,
            local_weights,
            curr_layer->inputs.rows * NUM_TEST_CASES,
            curr_layer->weights.rows + curr_layer->weights.align_pad,
            curr_layer->weights.cols,
            curr_layer->inputs.align_pad,
            curr_layer->activation,
            options->result_start,
            options->accumulate,
            local_results);


    if (curr_layer->output_req == IO_ACP ||
        curr_layer->output_req == IO_CACHE) {
        acp_pack_and_store_fp16(host_results,
                                local_results,
                                result_size,
                                options->dma_store_start,
                                options->dma_store_start);
    } else if (curr_layer->output_req == IO_DMA) {
        ASSERT(host_results && "DMA results pointer cannot be NULL!");
        dma_pack_and_store_fp16(host_results,
                                local_results,
                                result_size,
                                options->dma_store_start,
                                options->dma_store_start);
    }
}

void smv_inner_product_layer_hw(packed_fp16* dma_activations,
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
                                access_config* access_config,
                                smv_inner_product_options* options) {
    // We don't currently support using a local cache for inner products.  If
    // the IO requirement is IO_CACHE, it will be treated as IO_ACP.
    bool use_acp_results = (access_config->outputs == _ACP ||
                            access_config->outputs == _Cache);
    bool use_acp_inputs =
            (access_config->inputs == _ACP || access_config->inputs == _Cache);

    if (options->input_in_spad0) {
        if (use_acp_results) {
            if (use_acp_inputs) {
                smv_inner_product_layer_hw_impl(acp_activations, dma_weights,
                                                acp_results, spad0, umem, spad1,
                                                curr_layer, options);
            } else {
                // Not a common scenario but should be supported anyways.
                smv_inner_product_layer_hw_impl(dma_activations, dma_weights,
                                                acp_results, spad0, umem, spad1,
                                                curr_layer, options);
            }
        } else {
            if (use_acp_inputs) {
                // Common use case: Use ACP to load input, but leave data in
                // the spads.
                smv_inner_product_layer_hw_impl(acp_activations, dma_weights,
                                                dma_results, spad0, umem, spad1,
                                                curr_layer, options);
            } else {
                ASSERT((access_config->inputs == _DmaOrLocal &&
                        access_config->outputs == _DmaOrLocal) &&
                       "IO requirements are inconsistent with DMA fallback!");
                smv_inner_product_layer_hw_impl(dma_activations, dma_weights,
                                                dma_results, spad0, umem, spad1,
                                                curr_layer, options);
            }
        }
    } else {
        if (use_acp_results) {
            if (use_acp_inputs) {
                smv_inner_product_layer_hw_impl(acp_activations, dma_weights,
                                                acp_results, spad1, umem, spad0,
                                                curr_layer, options);
            } else {
                // Not a common scenario but should be supported anyways.
                smv_inner_product_layer_hw_impl(dma_activations, dma_weights,
                                                acp_results, spad1, umem, spad0,
                                                curr_layer, options);
            }
        } else {
            if (use_acp_inputs) {
                // Common use case: Use ACP to load input, but leave data in
                // the spads.
                smv_inner_product_layer_hw_impl(acp_activations, dma_weights,
                                                dma_results, spad1, umem, spad0,
                                                curr_layer, options);
            } else {
                ASSERT((access_config->inputs == _DmaOrLocal &&
                        access_config->outputs == _DmaOrLocal) &&
                       "IO requirements are inconsistent with DMA fallback!");
                smv_inner_product_layer_hw_impl(dma_activations, dma_weights,
                                                dma_results, spad1, umem, spad0,
                                                curr_layer, options);
            }
        }
    }
}

bool smv_inner_product_needs_work_division(layer_t* curr_layer,
                                           smv_global* g_smv) {
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    return total_weight_bytes > g_smv->kUmemSize;
}

// These are the conditions under which we just will not try to run the layer
// at all.
//
// Same as SMIV, but it might change.
void smv_inner_product_check_absolute_size_limits(layer_t* curr_layer,
                                                  smv_global* g_smv) {
    const unsigned total_input_bytes = get_input_activations_size(curr_layer) /
                                       NUM_TEST_CASES * sizeof(float);
    if (total_input_bytes > g_smv->kSpadSize) {
        printf("A single input does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    const unsigned total_output_bytes =
            get_output_activations_size(curr_layer) / NUM_TEST_CASES *
            sizeof(float);
    if (total_output_bytes > g_smv->kSpadSize) {
        printf("A single output does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
}

inner_product_tiling_cfg* init_inner_product_tiling_cfg(int num_tiles) {
    inner_product_tiling_cfg* cfg =
            (inner_product_tiling_cfg*)malloc(sizeof(inner_product_tiling_cfg));
    cfg->tiles =
            (inner_product_tile*)malloc(sizeof(inner_product_tile) * num_tiles);
    for (int i = 0; i < num_tiles; i++) {
        cfg->tiles[i].num = i;
    }
    cfg->num_tiles = num_tiles;
    return cfg;
}

inner_product_strip* init_inner_product_strips(int num_strips) {
    inner_product_strip* strips = (inner_product_strip*)malloc(
            sizeof(inner_product_strip) * num_strips);
    for (int i = 0; i < num_strips; i++)
        strips[i].num = i;
    return strips;
}

void free_inner_product_tiling_cfg(inner_product_tiling_cfg* cfg) {
    for (int tile = 0; tile < cfg->num_tiles; tile++) {
        free(cfg->tiles[tile].strips);
    }
    free(cfg->tiles);
    free(cfg);
}

void print_inner_product_tiling_cfg(inner_product_tiling_cfg* cfg) {
    INFO_MSG("Inner product tiling configuration\n");
    for (int t = 0; t < cfg->num_tiles; t++) {
        inner_product_tile* tile = &cfg->tiles[t];
        INFO_MSG("Tile %d\n", t);
        INFO_MSG("  Input starting offset: %d\n", tile->inputs_start_offset);
        for (int s = 0; s < tile->num_strips; s++) {
            inner_product_strip* strip = &tile->strips[s];
            INFO_MSG("  Strip %d\n", s);
            INFO_MSG("  - Weights starting row: %d\n",
                     strip->weights_start_row);
            INFO_MSG("  - Weights dims: %d x %d\n", strip->weights_dims[0],
                     strip->weights_dims[1]);
        }
    }
}

// Divides the work for a FC layer into several iterations on SMV.
//
// Work division is required when the number of weights exceeds what can be fit
// on the UMEM.
inner_product_tiling_cfg* smv_inner_product_tile_work(layer_t* curr_layer,
                                                      smv_global* g_smv) {
    smv_inner_product_check_absolute_size_limits(curr_layer, g_smv);
    if (!smv_inner_product_needs_work_division(curr_layer, g_smv)) {
        // Return a work configuration that just has one entry and contains the
        // whole weight matrix.
        inner_product_tiling_cfg* fc_cfg = init_inner_product_tiling_cfg(1);
        fc_cfg->tiles[0].strips = init_inner_product_strips(1);
        fc_cfg->tiles[0].strips[0].weights_dims[0] = curr_layer->weights.rows;
        fc_cfg->tiles[0].strips[0].weights_dims[1] = curr_layer->weights.cols;
        fc_cfg->tiles[0].strips[0].weights_start_row = 0;
        fc_cfg->tiles[0].inputs_start_offset = 0;
        fc_cfg->tiles[0].num_strips = 1;
        return fc_cfg;
    }

    /* Divide up the weights matrix in tiles and strips.
     *
     * Each strip should contain the maximum amount of work that can fit on the
     * accelerator at a single time. Each tile cannot exceed
     * max_inputs_per_tile columns.
     *
     * The weights are stored in col-major order.
     *
     *   Input activations ----->
     *
     *  output   [  T0  S0  ]  [  T1  S0  ]
     *  neurons  [  T0  S1  ]  [  T1  S1  ]
     *    |      [  T0  S2  ]  [  T1  S2  ]
     *    |      [  T0  S3  ]  [  T1  S3  ]
     *    v      [  T0  S4  ]  [  T1  S4  ]
     *           [  T0  S5  ]  [  T1  S5  ]
     */
    const int max_strip_size = g_smv->kUmemSize / sizeof(float);
    int num_inputs = curr_layer->weights.rows * NUM_TEST_CASES;
    int num_neurons = curr_layer->weights.cols + curr_layer->weights.align_pad;
    int max_inputs_per_tile = g_smv->kUmemSize / NUM_PE_INSTS / sizeof(float);
    int num_tiles = FRAC_CEIL(num_inputs, max_inputs_per_tile);
    inner_product_tiling_cfg* fc_cfg =
            init_inner_product_tiling_cfg(num_tiles);

    int inputs_remaining = num_inputs;
    for (int tile_num = 0; tile_num < num_tiles; tile_num++) {
        // Compute the size of each tile of weights and strip-mine each tile
        // in chunks of 8 output neurons.
        int num_tile_inputs = min2(inputs_remaining, max_inputs_per_tile);
        int max_neurons_per_strip =
                ((max_strip_size / num_tile_inputs) / NUM_PE_INSTS) *
                NUM_PE_INSTS;
        int num_strips = ceil(((float)num_neurons) / max_neurons_per_strip);
        inner_product_tile* tile = &fc_cfg->tiles[tile_num];
        tile->strips = init_inner_product_strips(num_strips);
        tile->num_strips = num_strips;
        tile->inputs_start_offset = (num_inputs - inputs_remaining);

        int num_neurons_remaining = num_neurons;
        for (int i = 0; i < num_strips; i++) {
            inner_product_strip* strip = &tile->strips[i];
            int num_neurons_this_iter =
                    min2(num_neurons_remaining, max_neurons_per_strip);
            // We can ignore align_pad here because num_neurons has already
            // accounted for the original required padding.
            strip->weights_dims[0] = num_tile_inputs;
            strip->weights_dims[1] = num_neurons_this_iter;
            strip->weights_start_row = i * max_neurons_per_strip;
            num_neurons_remaining -= num_neurons_this_iter;
        }
        inputs_remaining -= num_tile_inputs;
    }
    return fc_cfg;
}

void smv_run_eltwise_ops(data_list* host_activations,
                         layer_t* curr_layer,
                         bool input_in_spad0,
                         bool use_hw_activation_func) {
    layer_t eltwise_layer;
    eltwise_layer.type = curr_layer->type;
    eltwise_layer.num = curr_layer->num;
    eltwise_layer.inputs = curr_layer->inputs;
    eltwise_layer.outputs = curr_layer->outputs;
    eltwise_layer.weights = (dims_t){ 0, 0, 0, 0 };
    eltwise_layer.biases = (dims_t){ 1, curr_layer->outputs.cols, 1,
                                     curr_layer->outputs.align_pad };

    activation_type act_func = curr_layer->activation;
    bool do_hw_activation =
            use_hw_activation_func &&
            smiv_is_supported_activation_func(curr_layer->type, act_func);
    eltwise_layer.activation =
            do_hw_activation ? curr_layer->activation : NO_ACTIVATION;

    // The input req of eltwise is the output req of the layer's main kernel.
    // If the FC wanted to leave the inputs in the local scratchpad, we don't
    // want to DMA the host activations back.
    eltwise_layer.input_req = curr_layer->output_req;
    eltwise_layer.weights_req = IO_DMA;
    eltwise_layer.output_req = curr_layer->output_req;

    smv_eltwise_options options;
    options.do_bias = true;
    options.input_in_spad0 = input_in_spad0;

    data_list* host_weights = curr_layer->host_weights;
    int bias_index = host_weights->len - 1;
    fp16array_t* biases = host_weights->data[bias_index].dense_hp;
    fp16array_t* activations = host_activations->data[0].dense_hp;
    MAP_ARRAY_TO_ACCEL(g_smv.kInnerProductHw,
                       get_host_inputs_var_name(eltwise_layer.input_req),
                       activations->d, activations->size * sizeof(packed_fp16));
    MAP_ARRAY_TO_ACCEL(g_smv.kInnerProductHw,
                       get_host_weights_var_name(eltwise_layer.weights_req),
                       biases->d, biases->size * sizeof(packed_fp16));

    access_config access_config;
    access_config.inputs = io_to_access_mechanism(eltwise_layer.input_req);
    access_config.weights = io_to_access_mechanism(eltwise_layer.weights_req);
    access_config.outputs = io_to_access_mechanism(eltwise_layer.output_req);

    INVOKE_KERNEL_PROF(g_smv.kInnerProductHw,
                       eltwise_layer.num, smv_eltwise_hw,
                       activations->d, biases->d,  // DMA
                       activations->d, biases->d,  // Cache
                       activations->d, biases->d,  // ACP
                       g_smv.umem, g_smv.spad0, g_smv.spad1,
                       &eltwise_layer, &access_config, &options);
}

// Call the right HW function based on the device parameters.
void smv_inner_product_layer_hw_dispatch(packed_fp16* activations,
                                         packed_fp16* weights,
                                         packed_fp16* results,
                                         layer_t* layer,
                                         smv_global* g_smv,
                                         // Make a copy here.
                                         smv_inner_product_options options) {
    io_req_t input_req = layer->input_req;
    io_req_t weights_req = layer->weights_req;
    io_req_t output_req = layer->output_req;

    if (output_req != IO_NONE) {
        MAP_ARRAY_TO_ACCEL(
                g_smv->kInnerProductHw,
                get_host_results_var_name(output_req),
                results,
                get_nhwc_dims_size(&layer->outputs) * sizeof(float16));
    }
    // This needs to be handled separately from the inputs IO because if we
    // used compressed weights, then they have already been DMAed and
    // decompressed by the point we reach here.
    begin_ignored_profiling(layer->num);
    if (weights_req == IO_DMA) {
        int weights_size = get_num_weights_layer(layer, 0) * sizeof(float16);
        flush_cache_range(weights, weights_size);
    }
    if (input_req == IO_DMA || input_req == IO_NONE) {
        // Use DMA for weights/activations.
        // Flush cache lines for activations and weights.
        int activations_size =
                get_input_activations_size(layer) * sizeof(float16);
        flush_cache_range(activations, activations_size);
    }
    end_profiling();

    // This object is an internal structure only for the purposes of
    // simplifying the dispatch mechanism conditional checks!
    access_config access_config;
    access_config.inputs = io_to_access_mechanism(layer->input_req);
    access_config.weights = io_to_access_mechanism(layer->weights_req);
    access_config.outputs = io_to_access_mechanism(layer->output_req);
    INVOKE_KERNEL_PROF(g_smv->kInnerProductHw,
                       layer->num,
                       smv_inner_product_layer_hw,
                       // DMA
                       activations,
                       weights,
                       results,
                       // CACHE
                       activations,
                       weights,
                       results,
                       // ACP
                       activations,
                       weights,
                       results,
                       // Local scratchpads
                       g_smv->umem,
                       g_smv->spad0,
                       g_smv->spad1,
                       // Other options
                       layer,
                       &access_config,
                       &options);

}

// Decompress the weights if necessary.
//
// Although we implement CSR, we actually logically want CSC because the
// accelerator works on channel-last data.  We achieve the same effect by
// transposing the weights matrix prior to compression at the very beginning.
//
// Decompression will overwrite content stored in one of the scratchpads, so
// after decompression, the layer descriptor for the inner product needs to be
// updated to ensure partial sums are sent back to the host after every strip.
void smv_inner_product_run_decompression(
        layer_t partial_layer,
        packed_fp16* host_results,
        smv_decompression_options* decompress_options,
        smv_global* g_smv,
        device_t* device) {
    // Transpose the dimensions of the layer so we decompress the colmajor
    // weights correctly.
    int rows = partial_layer.weights.rows;
    partial_layer.weights.rows = partial_layer.weights.cols;
    partial_layer.weights.cols = rows;
    int weights_idx = decompress_options->tile_num;

    // Decompress the section of the CSR-formatted weights starting from
    // current_row, with the number of rows indicated in partial_layer.
    if (get_dims_size(&partial_layer.weights) > 0) {
        smiv_decompress_packed_csr_impl(&partial_layer,
                                        weights_idx,
                                        decompress_options->current_row,
                                        decompress_options->input_in_spad0,
                                        (smiv_global*)g_smv, device);
    }

    PRINT_MSG("Weights:\n");
    PRINT_DEBUG(g_smv->umem,
                partial_layer.weights.cols,
                partial_layer.weights.rows,
                partial_layer.weights.rows + partial_layer.weights.align_pad);
}

layer_t create_partial_layer(layer_t* curr_layer,
                             inner_product_tile* tile,
                             inner_product_strip* strip,
                             int num_tiles,
                             bool use_hw_activation_func) {
    layer_t partial_layer;

    // Copy the basic fields from curr_layer.
    partial_layer.type = curr_layer->type;
    partial_layer.num = curr_layer->num;
    partial_layer.activation = curr_layer->activation;
    partial_layer.host_weights = curr_layer->host_weights;
    partial_layer.stride = curr_layer->stride;
    partial_layer.pad = curr_layer->pad;
    partial_layer.pool = curr_layer->pool;
    partial_layer.input_preprocessing = curr_layer->input_preprocessing;

    // Set the dimensions of the inputs, weights, and outputs based on the tile
    // and strip.
    partial_layer.inputs.rows = NUM_TEST_CASES;
    partial_layer.inputs.cols = strip->weights_dims[0];
    partial_layer.inputs.height = 1;
    partial_layer.inputs.align_pad = 0;
    partial_layer.weights.rows = strip->weights_dims[0];
    partial_layer.weights.cols = strip->weights_dims[1];
    partial_layer.weights.height = 1;
    partial_layer.weights.align_pad =
            calc_padding(partial_layer.weights.rows, DATA_ALIGNMENT);
    partial_layer.outputs.rows = curr_layer->outputs.rows;
    partial_layer.outputs.cols = curr_layer->outputs.cols;
    partial_layer.outputs.height = 1;
    partial_layer.outputs.align_pad = 0;
    // We handle the biases as a separate eltwise operation. It is no longer
    // fused with the matrix multiply op.
    partial_layer.biases.rows = 0;
    partial_layer.biases.cols = 0;
    partial_layer.biases.height = 0;
    partial_layer.biases.align_pad = 0;

    partial_layer.input_req = strip->num == 0 ? curr_layer->input_req : IO_NONE;
    partial_layer.weights_req = curr_layer->weights_req;

    // The tiling strategy used for DMA: we accumulate all the output pixels on
    // the scratchpad iteration by iteration, and on the last strip of the last
    // tile, we copy them back to the host.
    bool is_last_strip = strip->num == (tile->num_strips - 1);
    bool is_last_tile = tile->num == (num_tiles - 1);
    partial_layer.output_req =
            is_last_tile && is_last_strip ? curr_layer->output_req : IO_NONE;
    return partial_layer;
}

void smv_inner_product_layer_impl_rowwise(data_list* host_activations,
                                          layer_t* curr_layer,
                                          data_list* host_results,
                                          smv_global* g_smv,
                                          device_t* device,
                                          bool input_in_spad0) {
    require_data_type(host_activations, 0, UncompressedHalfPrecision);
    assert(TRANSPOSE_WEIGHTS &&
           "SMV inner product requires transposed weights!");

    INFO_MSG("Running rowwise inner product.\n");
    begin_profiling("smv_inner_product_tile_work", curr_layer->num);
    inner_product_tiling_cfg* fc_cfgs =
            smv_inner_product_tile_work(curr_layer, g_smv);
    end_profiling();
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_inner_product_tiling_cfg(fc_cfgs);
    const size_t inputs_size = get_dims_size(&curr_layer->inputs) *
                               NUM_TEST_CASES * sizeof(float16);
    MAP_ARRAY_TO_ACCEL(g_smv->kInnerProductHw,
                       get_host_inputs_var_name(curr_layer->input_req),
                       host_activations->data[0].dense_hp->d,
                       inputs_size);

    for (int tile_num = 0; tile_num < fc_cfgs->num_tiles; tile_num++) {
        inner_product_tile* tile = &fc_cfgs->tiles[tile_num];

        fp16array_t* host_inputs_buffer;
        if (fc_cfgs->num_tiles > 1) {
            int num_inputs_in_tile = tile->strips[0].weights_dims[0];
            host_inputs_buffer =
                    init_fp16array(num_inputs_in_tile * NUM_TEST_CASES, false);
            copy_data_col_range(
                    (float*)host_activations->data[0].dense_hp->d,
                    &curr_layer->inputs,
                    tile->inputs_start_offset / 2,
                    host_inputs_buffer->size,
                    (float*)host_inputs_buffer->d);
        } else {
            host_inputs_buffer = host_activations->data[0].dense_hp;
        }

        // If decompression is required, and the offload mechanism is DMA, we
        // need to DMA output data back on every iteration. To ensure we don't
        // corrupt the host_results buffer due to cacheline alignment
        // requirements, we buffer the output data in separate place and then
        // copy the correct parts over.
        bool requires_decompression =
                (curr_layer->host_weights->type[tile_num] == PackedCSR);
        bool use_decomp_result_buf =
                requires_decompression && (curr_layer->output_req == IO_NONE ||
                                           curr_layer->output_req == IO_DMA);
        fp16array_t* decomp_result_buf = NULL;
        if (use_decomp_result_buf) {
            decomp_result_buf =
                    init_fp16array(tile->strips[0].weights_dims[1], true);
        }
        float16* curr_dense_weights_loc =
                (float16*)curr_layer->host_weights->data[tile_num].dense_hp->d;

        int current_row = 0;
        for (int strip_num = 0; strip_num < tile->num_strips; strip_num++) {
            inner_product_strip* strip = &tile->strips[strip_num];
            layer_t partial_layer = create_partial_layer(
                    curr_layer, tile, strip, fc_cfgs->num_tiles,
                    device->use_hw_activation_func);
            int weights_strip_size =
                    strip->weights_dims[0] * strip->weights_dims[1];
            PRINT_MSG("FC tile %d, strip %d: weights %dx%d\n", tile_num,
                      strip_num, partial_layer.weights.rows,
                      partial_layer.weights.cols);

            if (requires_decompression) {
                smv_decompression_options options;
                options.tile_num = tile_num;
                options.current_row = current_row;
                options.input_in_spad0 = input_in_spad0;
                smv_inner_product_run_decompression(
                        partial_layer, host_results->data[0].dense_hp->d,
                        &options, g_smv, device);
                curr_dense_weights_loc = NULL;  // To prevent us from DMAing it.

                // Now that we've decompressed the weights, we don't need to DMA
                // them again.
                partial_layer.weights_req = IO_NONE;

                // If decompression is needed, we'll also want to send back the
                // pre-activation function outputs.
                if (partial_layer.output_req == IO_NONE) {
                    if (curr_layer->output_req != IO_NONE)
                        partial_layer.output_req = curr_layer->output_req;
                    else
                        partial_layer.output_req = device->cpu_default_offload;
                }

                // Only send back the pixels produced by this strip.
                partial_layer.outputs.cols = strip->weights_dims[1];
            }

            if (partial_layer.weights_req != IO_NONE) {
                MAP_ARRAY_TO_ACCEL(g_smv->kInnerProductHw,
                                   get_host_weights_var_name(IO_DMA),
                                   curr_dense_weights_loc,
                                   weights_strip_size * sizeof(float16));
            }

            // This buffer is where the accelerator will store its data.
            packed_fp16* accel_result_loc =
                    use_decomp_result_buf ? decomp_result_buf->d
                                          : host_results->data[0].dense_hp->d;

            smv_inner_product_options options;
            options.input_in_spad0 = input_in_spad0;
            options.use_pipelined_dma = device->use_pipelined_dma;

            if (requires_decompression) {
                if (use_decomp_result_buf) {
                    // If we use the decompression result buffer, then we store
                    // back data on every strip, and it is always to the
                    // beginning of the buffer.
                    options.result_start = 0;
                    options.dma_store_start = 0;
                } else {
                    // If we don't use the decompression result buffer, then we
                    // store directly to the host result buffer, in which case
                    // we need to put it into the right place (the current row).
                    // The result_start could also be zero.
                    options.result_start = current_row;
                    options.dma_store_start = current_row;
                }
            } else {
                // If no decompression is needed, then we can accumulate the
                // partial sums on the private scratchpad over the strips and
                // only copy the results back on the very last iteration.
                // Therefore, the result_start must increment up (or we'd
                // overwrite the results of the previous strips), but when we
                // store the data back, it has to start at the start of the
                // host results buffer.
                options.result_start = current_row;
                options.dma_store_start = 0;
            }


            options.accumulate = tile_num > 0;
            ARRAY_2D(float16, _host_results, host_results->data[0].dense_hp->d,
                     curr_layer->outputs.cols + curr_layer->outputs.align_pad);
            if (options.accumulate && use_decomp_result_buf) {
                if (curr_layer->input_req != IO_NONE)
                    options.psums_req = curr_layer->input_req;
                else
                    options.psums_req = device->cpu_default_offload;
                for (int n = 0; n < NUM_TEST_CASES; n++) {
                    memcpy(accel_result_loc, &_host_results[n][current_row],
                           strip->weights_dims[1] * sizeof(float16));
                }
            } else {
                options.psums_req = IO_NONE;
            }

            smv_inner_product_layer_hw_dispatch(
                    host_inputs_buffer->d,
                    (packed_fp16*)curr_dense_weights_loc,
                    accel_result_loc,
                    &partial_layer,
                    g_smv,
                    options);

            // Copy the results from result_buf to host_results.
            if (use_decomp_result_buf) {
                for (int n = 0; n < NUM_TEST_CASES; n++) {
                    memcpy(&_host_results[n][current_row],
                           accel_result_loc + options.dma_store_start,
                           strip->weights_dims[1] * sizeof(float16));
                }
            }
            current_row += strip->weights_dims[1];
            curr_dense_weights_loc += weights_strip_size;
        }
        if (decomp_result_buf)
            free_fp16array(decomp_result_buf);

        if (fc_cfgs->num_tiles > 1) {
            free_fp16array(host_inputs_buffer);
        }
    }

    bool fc_result_in_spad0 = !input_in_spad0;
    smv_run_eltwise_ops(host_results, curr_layer, fc_result_in_spad0,
                        device->use_hw_activation_func);

    free_inner_product_tiling_cfg(fc_cfgs);
}

void smv_inner_product_layer_impl(data_list* host_activations,
                                  layer_t* layers,
                                  int lnum,
                                  data_list* host_results,
                                  smv_global* g_smv,
                                  device_t* device) {
    static float* current_result_loc = NULL;
    if (current_result_loc == NULL) {
        current_result_loc = g_smv->spad1;
    } else if (current_result_loc == g_smv->spad0) {
        current_result_loc = g_smv->spad1;
    } else if (current_result_loc == g_smv->spad1) {
        current_result_loc = g_smv->spad0;
    }
    bool input_in_spad0 = (current_result_loc == g_smv->spad1);
    layer_t* curr_layer = &layers[lnum];

    ASSERT(curr_layer->host_weights->type[0] != CSR &&
           "Unpacked CSR weights are not supported!");
    smv_inner_product_layer_impl_rowwise(host_activations,
                                         curr_layer,
                                         host_results,
                                         g_smv,
                                         device,
                                         input_in_spad0);
}
