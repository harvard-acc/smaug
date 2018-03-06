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

typedef struct _inner_product_options {
    bool do_bias;
    bool input_in_spad0;
    bool use_pipelined_dma;
    int result_start;
    int dma_store_start;
} smv_inner_product_options;

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
        // This size includes the biases if options->do_bias is true.
        int weights_size = get_num_weights_layer(curr_layer, 0);
        setReadyBits(local_weights, SMV_UMEM_SIZE, 0);
        dma_load_and_unpack_fp16(
                local_weights, host_weights, weights_size, 0, 0);
    }

    if (curr_layer->input_req == IO_DMA || curr_layer->input_req == IO_ACP ||
        curr_layer->input_req == IO_CACHE) {
        ASSERT(host_activations && "DMA inputs pointer cannot be NULL!");
        int activations_size = get_input_activations_size(curr_layer);
        if (curr_layer->input_req == IO_DMA) {
            setReadyBits(local_activations, SMV_SPAD_SIZE, 0);
            dma_load_and_unpack_fp16(local_activations,
                                     host_activations,
                                     activations_size, 0, 0);
        } else {
            acp_load_and_unpack_fp16(local_activations, host_activations,
                                     activations_size, 0, 0);
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
            local_results);

    VEC_ARRAY_1D(v8fp_t, _local_results, local_results);
    bool do_bias_or_activation =
            options->do_bias || curr_layer->activation != NO_ACTIVATION;
    if (do_bias_or_activation) {
        int output_cols = curr_layer->outputs.cols;
        VEC_ARRAY_1D(v8fp_t, _weights, local_weights);
        int bias_offset =
                (curr_layer->weights.cols *
                 (curr_layer->weights.rows + curr_layer->weights.align_pad)) /
                VECTOR_SIZE;
        const v8fp_t zero = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
        bias_and_act_func:
        for (int i = 0; i < FRAC_CEIL(output_cols, VECTOR_SIZE); i++) {
            v8fp_t psum = _local_results[i] +
                          (options->do_bias ? _weights[bias_offset + i] : zero);
            _local_results[i] =
                    activation_fun_simd_fxp(psum, curr_layer->activation);
        }
    }

    // If we didn't run the activation function or add the bias, and we expect
    // outputs to go out via ACP, then we need to explicitly store it back.
    size_t result_size = get_output_activations_size(curr_layer);
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

bool smv_inner_product_needs_work_division(layer_t* curr_layer) {
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    return total_weight_bytes > SMV_UMEM_SIZE;
}

// These are the conditions under which we just will not try to run the layer
// at all.
//
// Same as SMIV, but it might change.
void smv_inner_product_check_absolute_size_limits(layer_t* curr_layer) {
    const unsigned total_input_bytes = get_input_activations_size(curr_layer) /
                                       NUM_TEST_CASES * sizeof(float);
    if (total_input_bytes > SMIV_SPAD_SIZE) {
        printf("A single input does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    const unsigned total_output_bytes =
            get_output_activations_size(curr_layer) / NUM_TEST_CASES *
            sizeof(float);
    if (total_output_bytes > SMIV_SPAD_SIZE) {
        printf("A single output does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
}

// Divides the work for a FC layer into several iterations on SMV.
//
// Work division is required when the number of weights exceeds what can be fit
// on the UMEM. A col-major weight matrix is On x In, where In is the number of
// input neurons and On is the number of output neurons.
//
// Rowwise work division means to do the matrix multiply in groups of W x In,
// where W = On/iterations. This will require inputs reordering, but not
// weights reordering.
//
// **NOTE**: Unlike the SMIV tiling function, this does not add an additional
// row to fake the row of biases as a dimension on each iteration. Instead, on
// the last iteration, we set options.do_bias = True, so the last iteration
// will dmaLoad the biases, add them, and then run the activation function.
fc_cfg_t smv_inner_product_tile_rowwise(layer_t* curr_layer) {
    fc_cfg_t fc_cfgs;
    smv_inner_product_check_absolute_size_limits(curr_layer);
    if (!smv_inner_product_needs_work_division(curr_layer)) {
        // No work division means to return an fc_cfg_t that is holds the
        // entire weights.
        init_smiv_work_cfg(&fc_cfgs, 1);
        fc_cfgs.iteration[0] = curr_layer->weights;
        return fc_cfgs;
    }
    // Divide up the weights. The minimum amount of work is PEs x N, where N is
    // the number of input neurons. Also keep the bias in mind - it should be
    // omitted until the very last iteration.
    // If the final iteration has weights.rows == 1, then we know all that is
    // left to do is add the biases, so just do it in SW instead.
    const int num_inputs = curr_layer->weights.rows * NUM_TEST_CASES;
    const int num_neurons =
            curr_layer->weights.cols + curr_layer->weights.align_pad;
    const unsigned minimum_work_size =
            NUM_PE_INSTS * num_inputs * sizeof(float);
    if (minimum_work_size > SMV_UMEM_SIZE) {
        printf("This weights layer exceeds our current capability to run!\n");
        assert(false);
    }
    const unsigned max_work_units_per_iteration = SMV_UMEM_SIZE / minimum_work_size;
    const unsigned bytes_per_iteration =
            max_work_units_per_iteration * minimum_work_size;
    const unsigned num_rows_per_iteration =
            bytes_per_iteration / num_inputs / sizeof(float);
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    const unsigned num_iterations =
            ceil(((float)total_weight_bytes) / bytes_per_iteration);

    init_smiv_work_cfg(&fc_cfgs, num_iterations);
    unsigned num_rows_remaining = num_neurons;
    for (unsigned i = 0; i < num_iterations; i++) {
        int num_rows_this_iter =
                min2(num_rows_remaining, num_rows_per_iteration);
        // We can ignore align_pad here because num_neurons has already
        // accounted for the original required padding.
        fc_cfgs.iteration[i] = (dims_t){ num_inputs, num_rows_this_iter, 1, 0 };
        num_rows_remaining -= num_rows_this_iter;
    }
    return fc_cfgs;
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
// Biases have to be specified separately, because once we transpose, we lose
// the property that biases have the same padding and column widths as the
// weights. Since biases are usually not sparse, we don't compress them.
// Instead, we just directly DMA them into the accelerator.
//
// Finally, decompression will overwrite content stored in one of the
// scratchpads, so the layer descriptor for the inner product needs to be
// modified to account for this by doing some extra DMAs.
//
// The modified partial layer descriptor is updated directly via the first
// function argument.
void smv_inner_product_run_decompression_and_update_layer(
        layer_t* partial_layer,
        const layer_t* full_layer,
        dims_t* curr_iter,
        packed_fp16* host_results,
        int current_row,
        bool input_in_spad0,
        bool is_last_iter,
        smv_global* g_smv,
        device_t* device) {
    layer_t transpose_layer = *partial_layer;
    // Transpose the dimensions of the layer so we decompress the colmajor
    // weights correctly.
    int rows = transpose_layer.weights.rows;
    transpose_layer.weights.rows = transpose_layer.weights.cols;
    transpose_layer.weights.cols = rows;
    smiv_decompress_packed_csr_impl(&transpose_layer, 0, current_row,
                                    input_in_spad0, (smiv_global*)g_smv,
                                    device);
    if (is_last_iter) {
        assert(partial_layer->host_weights->len == 2 &&
               "Inner product HW on SMV must have two sets of "
               "weights!");
        assert(partial_layer->host_weights->type[1] == Uncompressed &&
               "The second set of weights (biases) must be "
               "uncompressed!");
        farray_t* biases = partial_layer->host_weights->data[1].dense;
        dma_options options;
        options.src_offset = 0;
        options.dst_offset = get_nhwc_dims_size(&transpose_layer.weights);
        options.use_pipelined_dma = device->use_pipelined_dma;
        options.length = biases->size * sizeof(float);
        options.is_load = true;
        dma_copy_impl(g_smv->umem, biases->d, g_smv->kInnerProductHw,
                      transpose_layer.num, g_smv, &options);
        // We need to copy the saved partial sums back to the appropriate
        // scratchpad. This needs to happen REGARDLESS of what the output
        // access mechanism is, because the inputs rely on DMA!
        // TODO: Allow the inner product kernel to pull in partial sums via
        // ACP.
        options.src_offset = 0;
        options.dst_offset = 0;
        options.use_pipelined_dma = device->use_pipelined_dma;
        options.length =
                next_multiple(current_row * sizeof(float16), CACHELINE_SIZE) *
                NUM_TEST_CASES;
        options.is_load = true;
        // TODO: THIS IS BROKEN - we can't just cast to float and call it a day
        // because DMA copy doesn't unpack from FP16 to FP32!
        dma_copy_impl(input_in_spad0 ? g_smv->spad1 : g_smv->spad0,
                      (float*)host_results, g_smv->kInnerProductHw,
                      transpose_layer.num, g_smv, &options);

        // On the last iteration, send the entire output back, because
        // only the last iteration applies the bias and activation
        // function, so don't change partial_layer->outputs.cols.
    } else {
        // Otherwise, send back only the pixels that were produced
        // during this iteration.
        partial_layer->outputs.cols = curr_iter->cols;
    }

    // Now that we've decompressed the weights, we don't need to DMA
    // them again.
    partial_layer->weights_req = IO_NONE;

    // If decompression is needed, we'll also want to send back the
    // pre-activation function outputs.
    if (partial_layer->output_req == IO_NONE)
        partial_layer->output_req = full_layer->output_req;

    PRINT_MSG("Weights:\n");
    PRINT_DEBUG(g_smv->umem,
                partial_layer->weights.cols,
                partial_layer->weights.rows,
                partial_layer->weights.rows + partial_layer->weights.align_pad);
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
    fc_cfg_t fc_cfgs = smv_inner_product_tile_rowwise(curr_layer);
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_smiv_work_cfg(&fc_cfgs);
    const size_t inputs_size = get_dims_size(&curr_layer->inputs) *
                               NUM_TEST_CASES * sizeof(float16);
    MAP_ARRAY_TO_ACCEL(g_smv->kInnerProductHw,
                       get_host_inputs_var_name(curr_layer->input_req),
                       host_activations->data[0].dense_hp->d,
                       inputs_size);

    int current_row = 0;
    bool requires_decompression =
            (curr_layer->host_weights->type[0] == PackedCSR);
    packed_fp16* curr_dense_weights_loc =
            curr_layer->host_weights->data[0].dense_hp->d;

    // If decompression is required, and the offload mechanism is DMA, we need
    // to DMA output data back on every iteration. To ensure we don't corrupt
    // the host_results buffer due to cacheline alignment requirements, we
    // buffer the output data in separate place and then copy the correct parts
    // over.
    bool use_decomp_result_buf =
            requires_decompression && (curr_layer->output_req == IO_NONE ||
                                       curr_layer->output_req == IO_DMA);
    fp16array_t* decomp_result_buf = NULL;
    if (use_decomp_result_buf) {
        decomp_result_buf = init_fp16array(fc_cfgs.iteration[0].cols, true);
    }

    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        layer_t partial_layer = *curr_layer;
        dims_t* curr_iter = &fc_cfgs.iteration[it];
        bool is_last_iter = (it == fc_cfgs.num_iterations - 1);
        bool do_bias = is_last_iter;

        // Optimization: On the last iteration, write directly to the final
        // host_results buffer, since we know we will copy the full output.
        bool use_decomp_result_buf_this_iter =
                (!is_last_iter && use_decomp_result_buf);
        // This buffer is where the accelerator will store its data.
        packed_fp16* accel_result_loc =
                use_decomp_result_buf_this_iter
                        ? decomp_result_buf->d
                        : host_results->data[0].dense_hp->d;

        // In this tiling strategy, generally, we accumulate all the output
        // pixels on the scratchpad iteration by iteration, and on the last
        // iteration we apply the biases and activation functions.
        if (is_last_iter) {
            // Check if we even support this activation function at all.
            activation_type act_func = partial_layer.activation;
            bool do_hw_activation = device->use_hw_activation_func &&
                                    smiv_is_supported_activation_func(
                                            partial_layer.type, act_func);
            if (!do_hw_activation)
                partial_layer.activation = NO_ACTIVATION;
            // Don't change the output_req.
        } else {
            // Don't run the activation function, and don't DMA back output
            // pixels.
            partial_layer.activation = NO_ACTIVATION;
            partial_layer.output_req = IO_NONE;
        }

        partial_layer.weights = *curr_iter;
        if (!do_bias) {
            partial_layer.biases.rows = 0;
            partial_layer.biases.cols = 0;
            partial_layer.biases.height = 0;
            partial_layer.biases.align_pad = 0;
        }

        int iter_weights_size =
                (curr_iter->rows) * (curr_iter->cols + curr_iter->align_pad);
        PRINT_MSG("FC iteration %d: weights %dx%d\n",
                   it,
                   partial_layer.weights.rows,
                   partial_layer.weights.cols);

        if (requires_decompression) {
            smv_inner_product_run_decompression_and_update_layer(
                    &partial_layer, curr_layer, curr_iter,
                    host_results->data[0].dense_hp->d, current_row,
                    input_in_spad0, is_last_iter, g_smv, device);
            curr_dense_weights_loc = NULL;  // To prevent us from DMAing it.
        }

        if (partial_layer.weights_req != IO_NONE) {
            const size_t weights_buffer_size =
                    (curr_iter->cols + curr_iter->align_pad) * curr_iter->rows *
                    sizeof(short);
            MAP_ARRAY_TO_ACCEL(
                    g_smv->kInnerProductHw,
                    get_host_weights_var_name(IO_DMA),
                    curr_dense_weights_loc,
                    weights_buffer_size);
        }

        smv_inner_product_options options;
        options.do_bias = do_bias;
        options.input_in_spad0 = input_in_spad0;
        options.use_pipelined_dma = device->use_pipelined_dma;
        // If we need to use the temporary buffer, start storing the outputs of
        // this tile from the beginning because it's only sized for a single
        // tile. Otherwise, start from current_row so we don't overwrite the
        // results of the previous tiles.
        options.result_start =
                use_decomp_result_buf_this_iter ? 0 : current_row;
        // On the last iteration, we want to DMA everything back.
        options.dma_store_start = is_last_iter ? 0 : options.result_start;
        smv_inner_product_layer_hw_dispatch(
                host_activations->data[0].dense_hp->d,
                curr_dense_weights_loc,
                accel_result_loc,
                &partial_layer,
                g_smv,
                options);

        PRINT_MSG_V("Partial results:\n");
        PRINT_DEBUG_V(input_in_spad0 ? g_smv->spad1 : g_smv->spad0,
                      curr_layer->inputs.rows * NUM_TEST_CASES,
                      curr_layer->outputs.cols,
                      curr_layer->outputs.cols + curr_layer->outputs.align_pad);

        // Copy the results from result_buf to host_results, unless it is the
        // last iteration (in which case the data is already there).
        if (use_decomp_result_buf_this_iter && !is_last_iter) {
            ARRAY_2D(float16, _host_results, host_results->data[0].dense_hp->d,
                     curr_layer->outputs.cols + curr_layer->outputs.align_pad);
            for (int n = 0; n < NUM_TEST_CASES; n++) {
                memcpy(&_host_results[n][current_row],
                       accel_result_loc + options.dma_store_start,
                       curr_iter->cols * sizeof(float16));
            }
        }

        current_row += curr_iter->cols;
        curr_dense_weights_loc += iter_weights_size / 2;
    }

    if (decomp_result_buf)
        free_fp16array(decomp_result_buf);
    free_smiv_work_cfg(&fc_cfgs);
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
