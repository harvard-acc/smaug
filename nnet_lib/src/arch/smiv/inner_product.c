#include <assert.h>
#include <string.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smiv/smiv.h"
#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/utility.h"
#include "config.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef void (*tiled_inner_product_impl)(
        float*, float*, layer_t*, float*, smiv_global*, device_t*, bool);

typedef struct _inner_product_options {
    bool do_bias;
    bool input_in_spad0;
    bool use_pipelined_dma;
} smiv_inner_product_options;

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
static void inner_product_layer_hw_impl(float* dma_activations,
                                        float* dma_weights,
                                        float* dma_results,
                                        float* local_activations,
                                        float* local_weights,
                                        float* local_results,
                                        layer_t* curr_layer,
                                        smiv_inner_product_options* options) {
    if (curr_layer->weights_req == IO_DMA) {
        ASSERT(dma_weights && "DMA weights pointer cannot be NULL!");
        int weights_size = get_num_weights_layer(curr_layer, 0);
        if (!options->do_bias)
            weights_size -= curr_layer->weights.cols;
        weights_size *= sizeof(float);
        setReadyBits(local_weights, weights_size, 0);
        dma_load_wrapper(local_weights, dma_weights, weights_size,
                         options->use_pipelined_dma);
    }

    if (curr_layer->input_req == IO_DMA) {
        ASSERT(dma_activations && "DMA inputs pointer cannot be NULL!");
        int activations_size =
                get_input_activations_size(curr_layer) * sizeof(float);
        setReadyBits(local_activations, activations_size, 0);
        dma_load_wrapper(local_activations, dma_activations, activations_size,
                         options->use_pipelined_dma);
    }

    matrix_multiply_with_bias_smiv(
            local_activations,
            local_weights,
            curr_layer->inputs.rows * NUM_TEST_CASES,
            curr_layer->weights.rows,
            curr_layer->weights.cols + curr_layer->weights.align_pad,
            curr_layer->inputs.align_pad,
            curr_layer->activation,
            options->do_bias,
            local_results);

    if (curr_layer->output_req == IO_DMA) {
        ASSERT(dma_results && "DMA results pointer cannot be NULL!");
        size_t result_size =
                get_output_activations_size(curr_layer) * sizeof(float);
        dma_store_wrapper(dma_results, local_results, result_size,
                          options->use_pipelined_dma);
    }
}

static void inner_product_layer_hw(float* dma_activations,
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
                                   smiv_inner_product_options* options) {

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
        inner_product_layer_hw_impl(NULL, NULL, NULL, INPUT##_activations,     \
                                    WGT##_weights, LR##_results, curr_layer,   \
                                    options);                                  \
    } while (0)

// DMA potentially used for all host arguments. The first three arguments are
// mechanism prefixes; all other arguments are the full variable names.
#define INNER_PROD_WITH_DMA_IMPL(HA, HW, HR, LA, LW, LR)                       \
    do {                                                                       \
        PRINT_MSG(#HA "-" #HW "-" #HR "-" #LA "-" #LW "-" #LR "\n");           \
        inner_product_layer_hw_impl(HA##_activations, HW##_weights,            \
                                    HR##_results, LA, LW, LR, curr_layer,      \
                                    options);                                  \
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

// Returns true if this inner product layer will require multiple iterations.
bool smiv_inner_product_needs_work_division(layer_t* curr_layer,
                                            smiv_global* g_smiv) {
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    return total_weight_bytes > g_smiv->kUmemSize;
}

// These are the conditions under which we just will not try to run the layer
// at all.
//
// TODO: These are not quite the right constraints.
void smiv_inner_product_check_absolute_size_limits(layer_t* curr_layer,
                                                   smiv_global* g_smiv) {
    const unsigned total_input_bytes =
            get_input_activations_size(curr_layer) / NUM_TEST_CASES;
    if (total_input_bytes > g_smiv->kSpadSize) {
        printf("A single input does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    const unsigned total_output_bytes =
            get_output_activations_size(curr_layer) / NUM_TEST_CASES;
    if (total_output_bytes > g_smiv->kSpadSize) {
        printf("A single output does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
}

// Divides the work for a FC layer into several iterations on SMIV.
//
// Work division is required when the number of weights exceeds what can be fit
// on the UMEM. The weight matrix is In x On, where In is the number of input
// neurons and On is the number of output neurons.
//
// Columnwise work division means to do the matrix multiply in groups of In x W,
// where W = On/iterations. This will require weights reordering, but not input
// reordering.
fc_cfg_t smiv_inner_product_divide_work_colwise(layer_t* curr_layer,
                                                smiv_global* g_smiv) {
    fc_cfg_t fc_cfgs;
    smiv_inner_product_check_absolute_size_limits(curr_layer, g_smiv);
    if (!smiv_inner_product_needs_work_division(curr_layer, g_smiv)) {
        // No work division means to return an fc_cfg_t that is holds the
        // entire weights.
        init_smiv_work_cfg(&fc_cfgs, 1);
        fc_cfgs.iteration[0] = curr_layer->weights;
        fc_cfgs.iteration[0].rows += curr_layer->biases.rows;
        return fc_cfgs;
    }
    // Divide up the weights. The minimum required work (for now) is an Nx8
    // strip of weights, where N is the number of hidden neurons.
    const int num_inputs = curr_layer->weights.rows + curr_layer->biases.rows;
    const unsigned num_neurons =
            curr_layer->weights.cols + curr_layer->weights.align_pad;
    const unsigned minimum_work_size = num_inputs * VECTOR_SIZE * sizeof(float);
    if (minimum_work_size > g_smiv->kUmemSize) {
        printf("This weights layer exceeds our current capability to run!\n");
        assert(false);
    }
    const unsigned max_work_units_per_iteration =
            g_smiv->kUmemSize / minimum_work_size;
    const unsigned bytes_per_iteration =
            max_work_units_per_iteration * minimum_work_size;
    const unsigned num_cols_per_iteration =
            bytes_per_iteration / num_inputs / sizeof(float);
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    const unsigned num_iterations =
            ceil(((float)total_weight_bytes) / bytes_per_iteration);

    init_smiv_work_cfg(&fc_cfgs, num_iterations);
    unsigned num_cols_remaining = num_neurons;
    for (unsigned i = 0; i < num_iterations; i++) {
        int num_cols_this_iter =
                min2(num_cols_remaining, num_cols_per_iteration);
        // We can ignore align_pad here because num_neurons has already
        // accounted for the original required padding.
        fc_cfgs.iteration[i] = (dims_t){ num_inputs, num_cols_this_iter, 1, 0 };
        num_cols_remaining -= num_cols_this_iter;
    }
    return fc_cfgs;
}

// Divides the work for a FC layer into several iterations on SMIV.
//
// Work division is required when the number of weights exceeds what can be fit
// on the UMEM. The weight matrix is In x On, where In is the number of input
// neurons and On is the number of output neurons.
//
// Rowwise work division means to do the matrix multiply in groups of W x On,
// where W = In/iterations. This will require inputs reordering, but not
// weights reordering.
fc_cfg_t smiv_inner_product_divide_work_rowwise(layer_t* curr_layer,
                                                smiv_global* g_smiv) {
    fc_cfg_t fc_cfgs;
    smiv_inner_product_check_absolute_size_limits(curr_layer, g_smiv);
    if (!smiv_inner_product_needs_work_division(curr_layer, g_smiv)) {
        // No work division means to return an fc_cfg_t that is holds the
        // entire weights.
        init_smiv_work_cfg(&fc_cfgs, 1);
        fc_cfgs.iteration[0] = curr_layer->weights;
        fc_cfgs.iteration[0].rows += curr_layer->biases.rows;
        return fc_cfgs;
    }
    // Divide up the weights. The minimum amount of work is 2xN, where N is the
    // number of output neurons. Also keep the bias in mind - it should be
    // omitted until the very last iteration.

    // num_inputs includes the extra row of biases. If the final iteration has
    // weights.rows == 1, then we know we should just add the biases in
    // software; otherwise we do it in HW.
    const int num_inputs =
            (curr_layer->weights.rows + curr_layer->biases.rows) *
            NUM_TEST_CASES;
    const int num_neurons =
            curr_layer->weights.cols + curr_layer->weights.align_pad;
    const unsigned minimum_work_size = num_neurons * 2 * sizeof(float);
    if (minimum_work_size > g_smiv->kUmemSize) {
        printf("This weights layer exceeds our current capability to run!\n");
        assert(false);
    }
    const unsigned max_work_units_per_iteration =
            g_smiv->kUmemSize / minimum_work_size;
    const unsigned bytes_per_iteration =
            max_work_units_per_iteration * minimum_work_size;
    const unsigned num_rows_per_iteration =
            bytes_per_iteration / num_neurons / sizeof(float);
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    const unsigned num_iterations =
            ceil(((float)total_weight_bytes) / bytes_per_iteration);

    init_smiv_work_cfg(&fc_cfgs, num_iterations);
    unsigned num_rows_remaining = num_inputs;
    for (unsigned i = 0; i < num_iterations; i++) {
        int num_rows_this_iter =
                min2(num_rows_remaining, num_rows_per_iteration);
        // We can ignore align_pad here because num_neurons has already
        // accounted for the original required padding.
        //
        // If this is not the last iteration, add one to the rows to fake a row
        // of biases (dimensionally). The bias will be skipped.
        bool is_last_iter = i == (num_iterations - 1);
        fc_cfgs.iteration[i] =
                (dims_t){ num_rows_this_iter + (is_last_iter ? 0 : 1),
                          num_neurons, 1, 0 };
        num_rows_remaining -= num_rows_this_iter;
    }
    return fc_cfgs;
}

// Call the right HW function based on the device parameters.
void smiv_inner_product_layer_hw_dispatch(float* activations,
                                          float* weights,
                                          float* results,
                                          layer_t* layer,
                                          int result_size,
                                          smiv_global* g_smiv,
                                          // Make a copy here.
                                          smiv_inner_product_options options) {
    io_req_t input_req = layer->input_req;
    io_req_t weights_req = layer->weights_req;
    io_req_t output_req = layer->output_req;

    if (output_req != IO_NONE) {
        MAP_ARRAY_TO_ACCEL(g_smiv->kInnerProductHw,
                           get_host_results_var_name(output_req),
                           results,
                           result_size * sizeof(float));
    }
    // This needs to be handled separately from the inputs IO because if we
    // used compressed weights, then they have already been DMAed and
    // decompressed by the point we reach here.
    if (weights_req == IO_DMA) {
        assert(weights && "Cannot DMA weights if weights don't exist!");
        begin_ignored_profiling(layer->num);
        int weights_size = get_num_weights_layer(layer, 0) * sizeof(float);
        flush_cache_range(weights, weights_size);
        end_profiling();
    }
    if (input_req == IO_DMA || input_req == IO_NONE) {
        // Use DMA for weights/activations.
        // Flush cache lines for activations and weights.
        begin_ignored_profiling(layer->num);
        int activations_size =
                get_input_activations_size(layer) * sizeof(float);
        flush_cache_range(activations, activations_size);
        end_profiling();
    }

    // This object is an internal structure only for the purposes of
    // simplifying the dispatch mechanism conditional checks!
    access_config access_config;
    access_config.inputs = io_to_access_mechanism(layer->input_req);
    access_config.weights = io_to_access_mechanism(layer->weights_req);
    access_config.outputs = io_to_access_mechanism(layer->output_req);
    INVOKE_KERNEL_PROF(g_smiv->kInnerProductHw,
                       layer->num,
                       inner_product_layer_hw,
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
                       g_smiv->umem,
                       g_smiv->spad0,
                       g_smiv->spad1,
                       // Other options
                       layer,
                       &access_config,
                       &options);

}

void smiv_inner_product_layer_impl_rowwise(float* host_activations,
                                           float* host_weights,
                                           layer_t* curr_layer,
                                           float* host_results,
                                           smiv_global* g_smiv,
                                           device_t* device,
                                           bool input_in_spad0) {
    INFO_MSG("Running rowwise inner product.\n");
    fc_cfg_t fc_cfgs =
            smiv_inner_product_divide_work_rowwise(curr_layer, g_smiv);
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_smiv_work_cfg(&fc_cfgs);
    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    bool do_bias_in_software =
            fc_cfgs.iteration[fc_cfgs.num_iterations - 1].rows == 1;
    if (do_bias_in_software) {
        // A NULL host_weights pointer is only passed if the weights are
        // compressed.
        assert(host_weights &&
               "Host weights cannot be NULL if bias is done in SW!");
    }

    // Holds a contiguous column of inputs and the partial results. If work
    // division is required, then each iteration's chunk of inputs is copied
    // into the buffer; otherwise, we just use the original input and results
    // buffers.
    float* host_inputs_buffer;
    float* host_results_buffer;
    const size_t inputs_buffer_size = curr_layer->inputs.rows *
                                      curr_layer->inputs.cols *
                                      NUM_TEST_CASES * sizeof(float);
    if (needs_multiple_iter) {
        host_inputs_buffer = (float*)malloc_aligned(inputs_buffer_size);
    } else {
        host_inputs_buffer = host_activations;
    }
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        host_results_buffer =
                (float*)malloc_aligned(get_dims_size(&curr_layer->outputs) *
                                       sizeof(float) * fc_cfgs.num_iterations);
    } else {
        host_results_buffer = host_results;
    }

    MAP_ARRAY_TO_ACCEL(g_smiv->kInnerProductHw,
                       get_host_inputs_var_name(curr_layer->input_req),
                       host_inputs_buffer,
                       inputs_buffer_size);

    int current_row = 0;
    float* current_result = host_results_buffer;
    float* curr_dense_weights_loc = host_weights;
    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        dims_t* curr_iter = &fc_cfgs.iteration[it];
        bool is_last_iter = (it == fc_cfgs.num_iterations - 1);
        bool do_bias = is_last_iter ? curr_iter->rows > 1 : false;
        // If the work division is such that at the end, we've done all the
        // multiplicative weights, and we only need to do the bias now, then we
        // just run the bias on the CPU.
        if (is_last_iter && curr_iter->rows == 1)
            break;

        layer_t partial_layer = *curr_layer;
        partial_layer.inputs.cols = curr_iter->rows - 1;
        partial_layer.weights = *curr_iter;
        partial_layer.outputs.cols = curr_iter->cols;
        partial_layer.biases = (dims_t){ 0, 0, 0, 0 };
        activation_type act_func = curr_layer->activation;
        bool do_hw_activation =
                device->use_hw_activation_func &&
                smiv_is_supported_activation_func(curr_layer->type, act_func);
        if (!do_hw_activation)
            partial_layer.activation = NO_ACTIVATION;

        int iter_weights_size = (curr_iter->rows - 1) *
                                (curr_iter->cols + curr_iter->align_pad);
        PRINT_MSG("FC iteration %d: weights %dx%d\n",
                   it,
                   partial_layer.weights.rows,
                   partial_layer.weights.cols);

        if (needs_multiple_iter) {
            copy_data_col_range(host_activations,
                                &curr_layer->inputs,
                                current_row,
                                curr_iter->rows - 1,
                                host_inputs_buffer);

            PRINT_MSG_V("inputs buffer\n");
            PRINT_DEBUG_V(host_inputs_buffer,
                          partial_layer.inputs.rows * NUM_TEST_CASES,
                          partial_layer.inputs.cols,
                          partial_layer.inputs.cols +
                                  partial_layer.inputs.align_pad);
            partial_layer.activation = NO_ACTIVATION;
        }

        // First decompress the weights.
        if (curr_layer->host_weights->type[0] == PackedCSR) {
            // If this is not the last iteration, then pre-emptively subtract
            // one from the rows to get rid of decompressing an extra row for
            // nothing.
            layer_t temp_layer = partial_layer;
            if (!is_last_iter)
                temp_layer.weights.rows--;
            smiv_decompress_packed_csr_impl(&temp_layer, 0, current_row,
                                            input_in_spad0, g_smiv, device);
            // Now that we've decompressed the weights, we don't need to DMA
            // them again.
            partial_layer.weights_req = IO_NONE;
            // Make sure we don't try to access host weights, since the
            // decompressed weights don't exist on the host.
            curr_dense_weights_loc = NULL;
            PRINT_MSG("Weights:\n");
            PRINT_DEBUG(g_smiv->umem,
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
                    g_smiv->kInnerProductHw,
                    get_host_weights_var_name(partial_layer.weights_req),
                    curr_dense_weights_loc,
                    weights_buffer_size);
        }
        size_t result_size = NUM_TEST_CASES * partial_layer.inputs.rows *
                             partial_layer.outputs.cols;

        smiv_inner_product_options options;
        options.do_bias = do_bias;
        options.input_in_spad0 = input_in_spad0;
        options.use_pipelined_dma = device->use_pipelined_dma;
        smiv_inner_product_layer_hw_dispatch(host_inputs_buffer,
                                             curr_dense_weights_loc,
                                             current_result,
                                             &partial_layer,
                                             result_size,
                                             g_smiv,
                                             options);

        PRINT_MSG_V("Partial results:\n");
        PRINT_DEBUG_V(current_result,
                      curr_layer->inputs.rows * NUM_TEST_CASES,
                      curr_iter->cols,
                      curr_iter->cols);

        current_row += curr_iter->rows - 1;
        current_result += result_size;
        curr_dense_weights_loc += iter_weights_size;
    }

    // If multiple iterations were needed, do a final round of reduction on the
    // partial sums.
    //
    // We have ITER blocks of NxM partial sums, where NxM is the final output
    // dimensions. Accumulate elementwise.
    //
    // TODO: For now, do on the CPU - but maybe using the reduction HW is
    // worthwhile?
    if (needs_multiple_iter) {
        // SMAUG expects this function to run the activation function if the HW
        // supports it and the user has not specified use_hw_activation_func =
        // false. But this particular flavor of dividing the inputs means that
        // unless we reduce in HW, we can't run activation functions in
        // hardware, since we never have the fully reduced sum there. As a
        // result, we have to run the activation function here, before we
        // return.
        activation_type act_func = curr_layer->activation;
        bool do_activation = act_func != NO_ACTIVATION;
        bool do_hw_activation =
                device->use_hw_activation_func &&
                smiv_is_supported_activation_func(curr_layer->type, act_func);
        bool do_activation_here = do_activation && do_hw_activation;

        int output_rows = curr_layer->outputs.rows;
        int output_cols =
                curr_layer->outputs.cols + curr_layer->outputs.align_pad;
        ARRAY_3D(float,
                 _temp_results,
                 host_results_buffer,
                 output_rows,
                 output_cols);  // temp buffer
        ARRAY_2D(float,
                 _host_results,
                 host_results,
                 output_cols);                 // dst buffer.
        float* biases = host_weights +
                        (curr_layer->weights.rows * curr_layer->weights.cols);
        current_result = host_results_buffer;  // temporary buffer.
        for (int r = 0; r < output_rows; r++) {
            for (int c = 0; c < output_cols; c++) {
                float accum = do_bias_in_software ? biases[c]: 0;
                for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
                  accum += _temp_results[it][r][c];
                }

                // If we need to do the activation function here, write it
                // straight back to temp_results, so we can use host_results
                // for the activation function results array.
                if (do_activation_here)
                  _temp_results[0][r][c] = accum;
                else
                  _host_results[r][c] = accum;
            }
        }
        if (do_activation_here) {
            // TODO: This means it will be harder to separate the MKL primitive
            // construction time from the actual activation function runtime.
            smiv_activation_function_impl(
                    host_results_buffer, curr_layer, host_results, device);
        }
    }

    if (needs_multiple_iter) {
        free(host_inputs_buffer);
    }
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        free(host_results_buffer);
    }

    free_smiv_work_cfg(&fc_cfgs);
}

void smiv_inner_product_layer_impl_colwise(float* host_activations,
                                           float* host_weights,
                                           layer_t* curr_layer,
                                           float* host_results,
                                           smiv_global* g_smiv,
                                           device_t* device,
                                           bool input_in_spad0) {
    INFO_MSG("Running colwise inner product.\n");
    fc_cfg_t fc_cfgs =
            smiv_inner_product_divide_work_colwise(curr_layer, g_smiv);
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_smiv_work_cfg(&fc_cfgs);

    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    MAP_ARRAY_TO_ACCEL(g_smiv->kInnerProductHw,
                       get_host_inputs_var_name(curr_layer->input_req),
                       host_activations,
                       INPUT_BYTES(curr_layer, 0));

    // Holds a contiguous column of weights and the partial results. If work
    // division is required, then each iteration's chunk of weights is copied
    // into the buffer; otherwise, we just use the original weights and results
    // buffers.
    float* host_weights_buffer;
    float* host_results_buffer;
    const size_t weights_buffer_size =
            (fc_cfgs.iteration[0].cols + fc_cfgs.iteration[0].align_pad) *
            fc_cfgs.iteration[0].rows * sizeof(float);
    if (needs_multiple_iter) {
        host_weights_buffer = (float*)malloc_aligned(weights_buffer_size);
    } else {
        host_weights_buffer = host_weights;
    }
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        host_results_buffer =
                (float*)malloc_aligned(OUTPUT_BYTES(curr_layer, 0));
    } else {
        host_results_buffer = host_results;
    }

    int current_col = 0;
    float* current_result = host_results_buffer;
    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        layer_t partial_layer = *curr_layer;
        dims_t* curr_iter = &fc_cfgs.iteration[it];
        partial_layer.weights = *curr_iter;
        partial_layer.biases = (dims_t){ 0, 0, 0, 0 };
        partial_layer.outputs.cols = curr_iter->cols;

        activation_type act_func = curr_layer->activation;
        bool do_hw_activation =
                device->use_hw_activation_func &&
                smiv_is_supported_activation_func(curr_layer->type, act_func);
        if (!do_hw_activation)
            partial_layer.activation = NO_ACTIVATION;

        PRINT_MSG("FC iteration %d: weights %dx%d\n",
                  it,
                  partial_layer.weights.rows,
                  partial_layer.weights.cols);

        if (needs_multiple_iter) {
            copy_data_col_range(host_weights,
                                &curr_layer->weights,
                                current_col,
                                curr_iter->cols + curr_iter->align_pad,
                                host_weights_buffer);

            PRINT_DEBUG_V(host_weights_buffer,
                          curr_iter->rows,
                          curr_iter->cols,
                          curr_iter->cols +
                                  curr_iter->align_pad);
        }

        if (curr_layer->weights_req != IO_NONE) {
            MAP_ARRAY_TO_ACCEL(
                    g_smiv->kInnerProductHw,
                    get_host_weights_var_name(curr_layer->weights_req),
                    host_weights_buffer,
                    weights_buffer_size);
        }

        size_t result_size = NUM_TEST_CASES * partial_layer.outputs.rows *
                             partial_layer.outputs.cols;

        smiv_inner_product_options options;
        options.do_bias = true;
        options.input_in_spad0 = input_in_spad0;
        options.use_pipelined_dma = device->use_pipelined_dma;
        smiv_inner_product_layer_hw_dispatch(host_activations,
                                             host_weights_buffer,
                                             current_result,
                                             &partial_layer,
                                             result_size,
                                             g_smiv,
                                             options);

        PRINT_MSG_V("Partial results:\n");
        PRINT_DEBUG_V(
                current_result,
                NUM_TEST_CASES,
                curr_iter->cols,
                curr_iter->cols + curr_iter->align_pad);

        current_col += curr_iter->cols;
        current_result += result_size;
    }

    // Fix up the results if required.
    //
    // The desired result looks like (for batch size 2):
    //
    // [ input 1, iter 1 results ] [ input 1, iter 2 results ] ...
    // [ input 1, iter 1 results ] [ input 1, iter 2 results ] ...
    //
    // But, when the batch size > 1 and multiple iterations are needed, the
    // results buffer will end up looking like this:
    //
    // [ input 1, iter 1 results ] [ input 2, iter 1 results ] ...
    // [ input 1, iter 2 results ] [ input 2, iter 2 results ] ...
    //
    // This routine reorders the results buffer and stores the result into the
    // final result array (host_results).
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        int output_size =
                curr_layer->outputs.cols + curr_layer->outputs.align_pad;
        ARRAY_2D(float, _host_results, host_results, output_size);  // dst buffer.
        current_result = host_results_buffer;  // temporary buffer.
        int curr_col = 0;
        for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
            dims_t* curr_iter = &fc_cfgs.iteration[it];
            int it_output_size = (curr_iter->cols +
                                  curr_iter->align_pad);
            for (int tc = 0; tc < NUM_TEST_CASES; tc++) {
                memcpy(&_host_results[tc][curr_col],
                       current_result,
                       it_output_size * sizeof(float));
                current_result += it_output_size;
            }
            curr_col += it_output_size;
        }
    }

    if (needs_multiple_iter) {
        free(host_weights_buffer);
    }
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        free(host_results_buffer);
    }

    free_smiv_work_cfg(&fc_cfgs);
}

void smiv_inner_product_layer_impl(float* host_activations,
                                   float* host_weights,
                                   layer_t* layers,
                                   int lnum,
                                   float* host_results,
                                   smiv_global* g_smiv,
                                   device_t* device) {
    static float* current_result_loc = NULL;
    if (current_result_loc == NULL) {
        current_result_loc = g_smiv->spad1;
    } else if (current_result_loc == g_smiv->spad0) {
        current_result_loc = g_smiv->spad1;
    } else if (current_result_loc == g_smiv->spad1) {
        current_result_loc = g_smiv->spad0;
    }
    bool input_in_spad0 = (current_result_loc == g_smiv->spad1);
    layer_t* curr_layer = &layers[lnum];
    assert(curr_layer->host_weights->len == 1 &&
           "SMIV only requires one set of weights!");

    if (curr_layer->host_weights->type[0] == Uncompressed) {
        float* host_weights_layer =
                (float*)curr_layer->host_weights->data[0].dense->d;
        PRINT_MSG("Weights:\n");
        PRINT_DEBUG(host_weights_layer,
                    curr_layer->weights.rows,
                    curr_layer->weights.cols,
                    curr_layer->weights.cols + curr_layer->weights.align_pad);

        // Dynamically pick one to use, based on whether weights or inputs are
        // better. If inputs is bigger, we'll divide the work columnwise in the
        // weights and reorder the weights instead of the inputs, which are
        // larger. But if weights are bigger, we'll divide the work rowwise in
        // the weights. This means we can just pass a pointer to the current
        // location in the weights and only reorder the inputs (the smaller
        // input to the GEMM).
        int input_size = get_input_activations_size(curr_layer);
        int weight_size = get_num_weights_layer(layers, lnum);
        INFO_MSG("Input size: %d, weight size: %d\n", input_size, weight_size);
        tiled_inner_product_impl impl =
                (input_size > weight_size)
                        ? &smiv_inner_product_layer_impl_colwise
                        : &smiv_inner_product_layer_impl_rowwise;
        impl(host_activations, host_weights_layer, curr_layer, host_results,
             g_smiv, device, input_in_spad0);
    } else if (curr_layer->host_weights->type[0] == PackedCSR) {
        // If the weights are stored in CSR format, we can only do row-wise
        // tiling.
        INFO_MSG("Running rowwise inner product for packed CSR weights.\n");
        smiv_inner_product_layer_impl_rowwise(host_activations,
                                              NULL,
                                              curr_layer,
                                              host_results,
                                              g_smiv,
                                              device,
                                              input_in_spad0);
    } else if (curr_layer->host_weights->type[0] == CSR) {
        fprintf(stderr, "Inner product layer for unpacked CSR weights is not "
                        "supported!\n");
        exit(1);
    }
}
