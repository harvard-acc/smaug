#include <assert.h>
#include <string.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
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
} smv_inner_product_options;

// Main implementation of inner product HW.
//
// This function handles DMA operations to load the local memory if required,
// runs the matrix multiply, and then handles the DMA transfer back to the host
// if required. In the context of this function, "local memory" is any type of
// memory that the accelerator can access directly (i.e. no DMA required). In
// other words, "local_activations" could be pointing to SPAD0, but it could
// have also been "acp_activations".
//
// For this reason, the first three arguments are prefixed with dma_, instead
// of a more generic prefix (since this wrapper is called with varying
// arguments), because they are only used if DMA is required.
//
// Arguments:
//   dma_activations: The host address of the input activations.
//   dma_weights: The host address of the input weights.
//   dma_results: The host address of the input results.
//   local_activations: Pointer to inputs that the accelerator reads directly.
//   local_weights: Pointer to weights that the accelerator reads directly.
//   local_results: Pointer to results that the accelerator writes directly.
//   curr_layer: Description of this layer's shape and parameters.
//   options: Additional options for this execution of inner product.
void smv_inner_product_layer_hw_impl(float* dma_activations,
                                     float* dma_weights,
                                     float* dma_results,
                                     float* local_activations,
                                     float* local_weights,
                                     float* local_results,
                                     layer_t* curr_layer,
                                     smv_inner_product_options* options) {
    if (curr_layer->weights_req == IO_DMA) {
        ASSERT(dma_weights && "DMA weights pointer cannot be NULL!");
        // This size includes the biases if options->do_bias is true.
        int weights_size = get_num_weights_layer(curr_layer, 0) * sizeof(float);
        setReadyBits(local_weights, SMV_UMEM_SIZE, 0);
        dma_load_wrapper(local_weights, dma_weights, weights_size,
                         options->use_pipelined_dma);
    }

    if (curr_layer->input_req == IO_DMA) {
        ASSERT(dma_activations && "DMA inputs pointer cannot be NULL!");
        int activations_size =
                get_input_activations_size(curr_layer) * sizeof(float);
        setReadyBits(local_activations, SMV_SPAD_SIZE, 0);
        dma_load_wrapper(local_activations, dma_activations, activations_size,
                         options->use_pipelined_dma);
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

    if (options->do_bias || curr_layer->activation != NO_ACTIVATION) {
        int output_cols = curr_layer->outputs.cols;
        VEC_ARRAY_1D(v8fp_t, _results, local_results);
        VEC_ARRAY_1D(v8fp_t, _weights, local_weights);
        int bias_offset =
                (curr_layer->weights.cols *
                 (curr_layer->weights.rows + curr_layer->weights.align_pad)) /
                VECTOR_SIZE;
        const v8fp_t zero = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int i = 0; i < FRAC_CEIL(output_cols, VECTOR_SIZE); i++) {
            v8fp_t psum = _results[i] +
                          (options->do_bias ? _weights[bias_offset + i] : zero);
            _results[i] = activation_fun_simd(psum, curr_layer->activation);
        }
    }

    if (curr_layer->output_req == IO_DMA) {
        ASSERT(dma_results && "DMA results pointer cannot be NULL!");
        size_t result_size =
                get_output_activations_size(curr_layer) * sizeof(float);
        dma_store_wrapper(dma_results, local_results, result_size,
                          options->use_pipelined_dma);
    }
}

void smv_inner_product_layer_hw(float* dma_activations,
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
                                layer_t* curr_layer,
                                access_config* access_config,
                                smv_inner_product_options* options) {

//=--------- Convenience macros for invoking the HW impl ---------------=//
//
// Each of these macros will call inner_product_layer_hw_impl() with a
// different name for the array arguments, based on the desired access
// mechanism for each. Since we name our variables in a consistent way -
// "mechanism_arrayname" - the macro can automatically form the correct
// variable name by macro concatentation.
//
// If DEBUG_LEVEL >= 2, then each invocation of these macros will print
// the mechanism and variable names used in the function call.
//
// Common argument abbreviations:
//    HA = host activations
//    HW = host weights
//    HR = host results
//    LA = local activations
//    LW = local weights
//    LR = local result

// No DMA involved, so we can pass NULL pointers for the first three arguments.
// All args to this macro are mechanism prefixes (e.g. dma, acp, cache).
#define INNER_PROD_NO_DMA_IMPL(INPUT, WGT, LR)                                 \
    do {                                                                       \
        PRINT_MSG(#INPUT "-" #WGT "-" #LR "\n");                               \
        smv_inner_product_layer_hw_impl(NULL, NULL, NULL, INPUT##_activations, \
                                        WGT##_weights, LR##_results,           \
                                        curr_layer, options);                  \
    } while (0)

// DMA potentially used for all host arguments. The first three arguments are
// mechanism prefixes; all other arguments are the full variable names.
#define INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, LA, LW, LR)                       \
    do {                                                                       \
        PRINT_MSG(#HA "-" #HW "-" #HR "-" #LA "-" #LW "-" #LR "\n");           \
        smv_inner_product_layer_hw_impl(HA##_activations, HW##_weights,        \
                                        HR##_results, LA, LW, LR, curr_layer,  \
                                        options);                              \
    } while (0)

// DMA used, with the input coming from either SPAD0 or SPAD1, but the output
// is going to a non-scratchpad location. Select the right input array with
// SELECT_SPAD0.
#define INNER_PROD_WITH_DMA_SPAD_INPUT_IMPL(                                   \
        HA, HW, HR, SPAD0, SPAD1, SELECT_SPAD0, LW, LR)                        \
    do {                                                                       \
        if (SELECT_SPAD0) {                                                    \
            INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, SPAD0, LW, LR);               \
        } else {                                                               \
            INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, SPAD1, LW, LR);               \
        }                                                                      \
    } while (0)

// DMA used, with the output going to either SPAD0 or SPAD1, but the input is
// coming from a non-scratchpad location. Select the right one with
// SELECT_SPAD0.
#define INNER_PROD_WITH_DMA_SPAD_OUTPUT_IMPL(                                  \
        HA, HW, HR, LA, LW, SPAD0, SPAD1, SELECT_SPAD0)                        \
    do {                                                                       \
        if (SELECT_SPAD0) {                                                    \
            INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, LA, LW, SPAD1);               \
        } else {                                                               \
            INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, LA, LW, SPAD0);               \
        }                                                                      \
    } while (0)

// DMA used, with both inputs and outputs going to/from scratchpads.
#define INNER_PROD_WITH_DMA_SPAD_IO_IMPL(                                      \
        HA, HW, HR, LW, SPAD0, SPAD1, SELECT_SPAD0)                            \
    do {                                                                       \
        if (SELECT_SPAD0) {                                                    \
            INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, SPAD0, LW, SPAD1);            \
        } else {                                                               \
            INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, SPAD1, LW, SPAD0);            \
        }                                                                      \
    } while (0)

    bool input_in_spad0 = options->input_in_spad0;
    // These selections use the same mechanism all across.
    if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _DmaOrLocal)) {
        INNER_PROD_WITH_DMA_SPAD_IO_IMPL(
            dma, dma, dma, umem, spad0, spad1, input_in_spad0);
    } else if (DISPATCH_3(access_config, _ACP, _ACP, _ACP)) {
        INNER_PROD_NO_DMA_IMPL(acp, acp, acp);
    } else if (DISPATCH_3(access_config, _Cache, _Cache, _Cache)) {
        INNER_PROD_NO_DMA_IMPL(cache, cache, cache);
    }
    // These selections only use _ACP or _Cache for the results.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _ACP)) {
        INNER_PROD_WITH_DMA_SPAD_INPUT_IMPL(
                dma, dma, acp, spad0, spad1, input_in_spad0, umem, acp_results);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _Cache)) {
        INNER_PROD_WITH_DMA_SPAD_INPUT_IMPL(
                dma, dma, cache, spad0, spad1, input_in_spad0, umem, cache_results);
    }
    // These selections use DMA/None for the inputs.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _ACP, _ACP)) {
        INNER_PROD_WITH_DMA_SPAD_INPUT_IMPL(dma, acp, acp, spad0, spad1,
                                            input_in_spad0, acp_weights,
                                            acp_results);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _Cache, _Cache)) {
        INNER_PROD_WITH_DMA_SPAD_INPUT_IMPL(dma, cache, cache, spad0, spad1,
                                            input_in_spad0, cache_weights,
                                            cache_results);
    }
    // These selections use DMA/None for the inputs/outputs.
    //
    // NOTE: This scenario is currently not possible to specify via the model
    // configuration file.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _ACP, _DmaOrLocal)) {
        INNER_PROD_WITH_DMA_SPAD_IO_IMPL(
            dma, acp, dma, acp_weights, spad0, spad1, input_in_spad0);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _Cache, _DmaOrLocal)) {
        INNER_PROD_WITH_DMA_SPAD_IO_IMPL(
            dma, cache, dma, cache_weights, spad0, spad1, input_in_spad0);
    }
    // These selections use DMA/None for the outputs.
    //
    // TODO: Since we're not reading out of the scratchpads for the input, it
    // shouldn't matter which scratchpad we write into, but currently, the
    // inner product layer will automatically toggle input_in_spad0 back and
    // forth regardless of what actually happened. To ensure that future layers
    // will get the right data, we still have to obey this condition. This
    // needs to be fixed.
    else if (DISPATCH_3(access_config, _ACP, _ACP, _DmaOrLocal)) {
        INNER_PROD_WITH_DMA_SPAD_OUTPUT_IMPL(acp, acp, dma, acp_activations,
                                             acp_weights, spad0, spad1,
                                             input_in_spad0);
    } else if (DISPATCH_3(access_config, _Cache, _Cache, _DmaOrLocal)) {
        INNER_PROD_WITH_DMA_SPAD_OUTPUT_IMPL(cache, cache, dma,
                                             cache_activations, cache_weights,
                                             spad0, spad1, input_in_spad0);
    }
    // These selections only use DMA for the weights.
    // This is used if the FC layer is using decompressed CSR weights (after
    // decompression the weights are already in the UMEM so no more data
    // movement is required).
    else if (DISPATCH_3(access_config, _ACP, _DmaOrLocal, _ACP)) {
        INNER_PROD_WITH_DMA_IMPL(
                acp, dma, acp, acp_activations, umem, acp_results);
    } else if (DISPATCH_3(access_config, _Cache, _DmaOrLocal, _Cache)) {
        INNER_PROD_WITH_DMA_IMPL(
                cache, dma, cache, cache_activations, umem, cache_results);
    }
    // Otherwise, give up.
    else {
        assert(false &&
               "This is an unsupported combination of access mechanisms!");
    }

#undef INNER_PROD_WITH_DMA_SPAD_INPUT_IMPL
#undef INNER_PROD_WITH_DMA_SPAD_OUTPUT_IMPL
#undef INNER_PROD_WITH_DMA_SPAD_IO_IMPL
#undef INNER_PROD_WITH_DMA_IMPL
#undef INNER_PROD_NO_DMA_IMPL
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
    const unsigned total_input_bytes =
            get_input_activations_size(curr_layer) / NUM_TEST_CASES;
    if (total_input_bytes > SMIV_SPAD_SIZE) {
        printf("A single input does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    const unsigned total_output_bytes =
            get_output_activations_size(curr_layer) / NUM_TEST_CASES;
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
void smv_inner_product_layer_hw_dispatch(float* activations,
                                         float* weights,
                                         float* results,
                                         layer_t* layer,
                                         int result_size,
                                         smv_global* g_smv,
                                         // Make a copy here.
                                         smv_inner_product_options options) {
    io_req_t input_req = layer->input_req;
    io_req_t weights_req = layer->weights_req;
    io_req_t output_req = layer->output_req;

    if (output_req != IO_NONE) {
        MAP_ARRAY_TO_ACCEL(kSmvInnerProductHw,
                           get_host_results_var_name(output_req),
                           results,
                           result_size * sizeof(float));
    }
    // This needs to be handled separately from the inputs IO because if we
    // used compressed weights, then they have already been DMAed and
    // decompressed by the point we reach here.
    if (weights_req == IO_DMA) {
        begin_ignored_profiling(layer->num);
        int weights_size = get_num_weights_layer(layer, 0);
        flush_cache_range(weights, weights_size);
        end_profiling();
    }
    if (input_req == IO_DMA || input_req == IO_NONE) {
        // Use DMA for weights/activations.
        // Flush cache lines for activations and weights.
        begin_ignored_profiling(layer->num);
        int activations_size = get_input_activations_size(layer);
        flush_cache_range(activations, activations_size);
        end_profiling();
    }

    // This object is an internal structure only for the purposes of
    // simplifying the dispatch mechanism conditional checks!
    access_config access_config;
    access_config.inputs = io_to_access_mechanism(layer->input_req);
    access_config.weights = io_to_access_mechanism(layer->weights_req);
    access_config.outputs = io_to_access_mechanism(layer->output_req);
    INVOKE_KERNEL_PROF(kSmvInnerProductHw,
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

void smv_inner_product_layer_impl_rowwise(float* host_activations,
                                          float* host_weights,
                                          layer_t* curr_layer,
                                          float* host_results,
                                          smv_global* g_smv,
                                          device_t* device,
                                          bool input_in_spad0) {
    assert(TRANSPOSE_WEIGHTS &&
           "SMV inner product requires transposed weights!");

    INFO_MSG("Running rowwise inner product.\n");
    fc_cfg_t fc_cfgs = smv_inner_product_tile_rowwise(curr_layer);
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_smiv_work_cfg(&fc_cfgs);
    const size_t inputs_size = curr_layer->inputs.rows *
                                      curr_layer->inputs.cols *
                                      NUM_TEST_CASES * sizeof(float);

    MAP_ARRAY_TO_ACCEL(kSmvInnerProductHw,
                       get_host_inputs_var_name(curr_layer->input_req),
                       host_activations,
                       inputs_size);

    int current_row = 0;
    float* curr_dense_weights_loc = host_weights;
    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        dims_t* curr_iter = &fc_cfgs.iteration[it];
        bool is_last_iter = (it == fc_cfgs.num_iterations - 1);
        bool do_bias = is_last_iter;

        layer_t partial_layer = *curr_layer;
        // If this is not the last iteration, we don't want to run the
        // activation function, and we don't want to DMA the data back until
        // the end.
        if (!is_last_iter) {
            partial_layer.activation = NO_ACTIVATION;
            if (partial_layer.output_req == IO_DMA)
                partial_layer.output_req = IO_NONE;
        } else {
            // Check if we even support this activation function at all.
            activation_type act_func = partial_layer.activation;
            bool do_hw_activation = device->use_hw_activation_func &&
                                    smiv_is_supported_activation_func(
                                            partial_layer.type, act_func);
            if (!do_hw_activation)
                partial_layer.activation = NO_ACTIVATION;
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

        // First decompress the weights.
        //
        // Although this is implementing CSR, we actually want CSC, logically.
        // We achieve the same effect by transposing the weights matrix prior
        // to compression.
        if (curr_layer->wgt_storage_type == PackedCSR) {
            smiv_decompress_packed_csr_impl(&partial_layer, current_row,
                                            input_in_spad0, (smiv_global*)g_smv,
                                            device);
            // Now that we've decompressed the weights, we don't need to DMA
            // them again.
            partial_layer.weights_req = IO_NONE;
            PRINT_MSG("Weights:\n");
            PRINT_DEBUG(g_smv->umem,
                        partial_layer.weights.rows,
                        partial_layer.weights.cols,
                        partial_layer.weights.cols +
                                partial_layer.weights.align_pad);
        }

        if (partial_layer.weights_req != IO_NONE) {
            const size_t weights_buffer_size =
                    (curr_iter->cols + curr_iter->align_pad) * curr_iter->rows *
                    sizeof(float);
            MAP_ARRAY_TO_ACCEL(
                    kSmvInnerProductHw,
                    get_host_weights_var_name(partial_layer.weights_req),
                    curr_dense_weights_loc,
                    weights_buffer_size);
        }
        size_t result_size = NUM_TEST_CASES * partial_layer.inputs.rows *
                             partial_layer.outputs.cols;

        smv_inner_product_options options;
        options.do_bias = do_bias;
        options.input_in_spad0 = input_in_spad0;
        options.use_pipelined_dma = device->use_pipelined_dma;
        options.result_start = current_row;
        smv_inner_product_layer_hw_dispatch(host_activations,
                                            curr_dense_weights_loc,
                                            host_results,
                                            &partial_layer,
                                            result_size,
                                            g_smv,
                                            options);

        PRINT_MSG_V("Partial results:\n");
        PRINT_DEBUG_V(host_results,
                      curr_layer->inputs.rows * NUM_TEST_CASES,
                      curr_iter->cols,
                      curr_iter->cols);

        current_row += curr_iter->cols;
        curr_dense_weights_loc += iter_weights_size;
    }

    free_smiv_work_cfg(&fc_cfgs);
}

void smv_inner_product_layer_impl(float* host_activations,
                                  float* host_weights,
                                  layer_t* layers,
                                  int lnum,
                                  float* host_results,
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
    float* host_weights_layer = (float*)curr_layer->host_weights_buffer;

    smv_inner_product_layer_impl_rowwise(host_activations,
                                         host_weights_layer,
                                         curr_layer,
                                         host_results,
                                         g_smv,
                                         device,
                                         input_in_spad0);
}
