#include <assert.h>
#include <string.h>

#include "arch/smiv_common.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smiv/smiv.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

void inner_product_layer_hw_impl(float* host_activations,
                                 float* host_weights,
                                 float* local_weights,
                                 float* local_inputs,
                                 float* results,
                                 layer_t* all_layers,
                                 int lnum,
                                 bool do_bias) {
    activation_type act_func = all_layers[lnum].activation;
    bool run_activation = act_func == RELU || act_func == RELU_THRESHOLD;
    int weights_size = get_num_weights_layer(all_layers, lnum);
    if (!do_bias)
        weights_size -= all_layers[lnum].weights.cols;
    weights_size *= sizeof(float);
    setReadyBits(local_weights, UMEM_SIZE, 0);
    dmaLoad(local_weights, host_weights, weights_size);

    if (all_layers[lnum].input_req == IO_DMA) {
        grab_input_activations_dma(
                host_activations, local_inputs, &all_layers[lnum]);
    }

    matrix_multiply_with_bias_smiv(
            local_inputs,
            local_weights,
            all_layers[lnum].inputs.rows * NUM_TEST_CASES,
            all_layers[lnum].weights.rows,
            all_layers[lnum].weights.cols + all_layers[lnum].weights.align_pad,
            all_layers[lnum].inputs.align_pad,
            run_activation,
            do_bias,
            results);
}

// HW accelerated inner product, using DMA for data movement.
//
// All arguments prefixed with host_ are host memory pointers and can only be
// deferenced from the host, except when performing a DMA operation.
void inner_product_layer_hw(float* host_activations,
                            float* host_weights,
                            float* umem,
                            float* spad0,
                            float* spad1,
                            layer_t* all_layers,
                            int lnum,
                            bool input_in_spad0,
                            bool do_bias,
                            float* host_result) {
    bool output_dma_req = (all_layers[lnum].output_req == IO_DMA);
    if (input_in_spad0) {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad0,
                                    spad1,
                                    all_layers,
                                    lnum,
                                    do_bias);
        if (output_dma_req)
            store_output_activations_dma(host_result, spad1, &all_layers[lnum]);
    } else {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad1,
                                    spad0,
                                    all_layers,
                                    lnum,
                                    do_bias);
        if (output_dma_req)
            store_output_activations_dma(host_result, spad0, &all_layers[lnum]);
    }
}

// HW accelerated inner product, using ACP for output data movement.
//
// All arguments prefixed with host_ are host memory pointers. acp_result is
// the host result pointer to be accessed over ACP.
void inner_product_layer_acp_hw(float* host_activations,
                                float* host_weights,
                                float* acp_result,
                                float* umem,
                                float* spad0,
                                float* spad1,
                                layer_t* all_layers,
                                int lnum,
                                bool input_in_spad0,
                                bool do_bias) {
    if (input_in_spad0) {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad0,
                                    acp_result,
                                    all_layers,
                                    lnum,
                                    do_bias);
    } else {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad1,
                                    acp_result,
                                    all_layers,
                                    lnum,
                                    do_bias);
    }
}

// Returns true if this inner product layer will require multiple iterations.
bool inner_product_needs_work_division(layer_t* curr_layer) {
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    return total_weight_bytes > UMEM_SIZE;
}

// These are the conditions under which we just will not try to run the layer
// at all.
//
// TODO: These are not quite the right constraints.
void check_absolute_size_limits(layer_t* curr_layer) {
    const unsigned total_input_bytes =
            get_input_activations_size(curr_layer) / NUM_TEST_CASES;
    if (total_input_bytes > SPAD_SIZE) {
        printf("A single input does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    const unsigned total_output_bytes =
            get_output_activations_size(curr_layer) / NUM_TEST_CASES;
    if (total_output_bytes > SPAD_SIZE) {
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
fc_cfg_t inner_product_divide_work_colwise(layer_t* curr_layer) {
    fc_cfg_t fc_cfgs;
    check_absolute_size_limits(curr_layer);
    if (!inner_product_needs_work_division(curr_layer)) {
        // No work division means to return an fc_cfg_t that is holds the
        // entire weights.
        init_work_cfg(&fc_cfgs, 1);
        fc_cfgs.iteration[0] = curr_layer->weights;
        return fc_cfgs;
    }
    // Divide up the weights. The minimum required work (for now) is an Nx8
    // strip of weights, where N is the number of hidden neurons.
    const int num_inputs = curr_layer->weights.rows;
    const unsigned num_neurons =
            curr_layer->weights.cols + curr_layer->weights.align_pad;
    const unsigned minimum_work_size = num_inputs * VECTOR_SIZE * sizeof(float);
    if (minimum_work_size > UMEM_SIZE) {
        printf("This weights layer exceeds our current capability to run!\n");
        assert(false);
    }
    const unsigned max_work_units_per_iteration = UMEM_SIZE / minimum_work_size;
    const unsigned bytes_per_iteration =
            max_work_units_per_iteration * minimum_work_size;
    const unsigned num_cols_per_iteration =
            bytes_per_iteration / num_inputs / sizeof(float);
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    const unsigned num_iterations =
            ceil(((float)total_weight_bytes) / bytes_per_iteration);

    init_work_cfg(&fc_cfgs, num_iterations);
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
fc_cfg_t inner_product_divide_work_rowwise(layer_t* curr_layer) {
    fc_cfg_t fc_cfgs;
    check_absolute_size_limits(curr_layer);
    if (!inner_product_needs_work_division(curr_layer)) {
        // No work division means to return an fc_cfg_t that is holds the
        // entire weights.
        init_work_cfg(&fc_cfgs, 1);
        fc_cfgs.iteration[0] = curr_layer->weights;
        return fc_cfgs;
    }
    // Divide up the weights. The minimum amount of work is 2xN, where N is the
    // number of output neurons. Also keep the bias in mind - it should be
    // omitted until the very last iteration.

    // num_inputs includes the extra row of biases. If the final iteration has
    // weights.rows == 1, then we know we should just add the biases in
    // software; otherwise we do it in HW.
    const int num_inputs = curr_layer->weights.rows * NUM_TEST_CASES;
    const int num_neurons =
            curr_layer->weights.cols + curr_layer->weights.align_pad;
    const unsigned minimum_work_size = num_neurons * 2 * sizeof(float);
    if (minimum_work_size > UMEM_SIZE) {
        printf("This weights layer exceeds our current capability to run!\n");
        assert(false);
    }
    const unsigned max_work_units_per_iteration = UMEM_SIZE / minimum_work_size;
    const unsigned bytes_per_iteration =
            max_work_units_per_iteration * minimum_work_size;
    const unsigned num_rows_per_iteration =
            bytes_per_iteration / num_neurons / sizeof(float);
    const unsigned total_weight_bytes = WEIGHT_BYTES(curr_layer, 0);
    const unsigned num_iterations =
            ceil(((float)total_weight_bytes) / bytes_per_iteration);

    init_work_cfg(&fc_cfgs, num_iterations);
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

// Copy a range of columns from a 2D array data buffer to a new buffer.
//
// The data from section starting at (row, col) = (0, start_col) to (num_rows,
// start_col + num_cols) will be copied.
//
// Args:
//   original_data: Original data buffer
//   original_dims: Dimensions of this buffer. Height is ignored.
//   start_col: The starting column.
//   num_cols: Number of cols in the range to copy.
//   new_buffer: Destination buffer.
void copy_data_col_range(float* original_data,
                         dims_t* original_dims,
                         int start_col,
                         int num_cols,
                         float* new_buffer) {
    int num_rows = original_dims->rows * NUM_TEST_CASES;
    int num_total_cols =
            original_dims->cols + original_dims->align_pad;
    ARRAY_2D(float, _data, original_data, num_total_cols);
    for (int r = 0; r < num_rows; r++) {
        memcpy(new_buffer + r * num_cols,
               &_data[r][start_col],
               num_cols * sizeof(float));
    }
}

// Call the right HW function based on the device parameters.
void inner_product_layer_hw_dispatch(float* activations,
                                     float* weights,
                                     layer_t* layer,
                                     bool input_in_spad0,
                                     bool do_bias,
                                     float* results,
                                     int result_size,
                                     device_t* device) {
    bool use_acp_offload = (device->cpu_activation_func_offload == IO_ACP);
    if (use_acp_offload) {
        MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                           "acp_result",
                           results,
                           result_size * sizeof(float));
        INVOKE_KERNEL_PROF(kInnerProductHw,
                           layer->num,
                           inner_product_layer_acp_hw,
                           activations,
                           weights,
                           results,
                           g_umem,
                           g_spad0,
                           g_spad1,
                           layer,
                           0,
                           input_in_spad0,
                           do_bias);
    } else {
        MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                           "host_result",
                           results,
                           result_size * sizeof(float));
        INVOKE_KERNEL_PROF(kInnerProductHw,
                           layer->num,
                           inner_product_layer_hw,
                           activations,
                           weights,
                           g_umem,
                           g_spad0,
                           g_spad1,
                           layer,
                           0,
                           input_in_spad0,
                           do_bias,
                           results);
    }
}

void inner_product_layer_impl_rowwise(float* host_activations,
                                      float* host_weights,
                                      layer_t* curr_layer,
                                      float* host_result,
                                      device_t* device,
                                      bool input_in_spad0) {
    fc_cfg_t fc_cfgs = inner_product_divide_work_rowwise(curr_layer);
    printf("Inner product layer %d work configuration:\n", curr_layer->num);
    print_work_cfg(&fc_cfgs);
    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    bool do_bias_in_software =
            fc_cfgs.iteration[fc_cfgs.num_iterations - 1].rows == 1;
    bool run_activation_in_software = needs_multiple_iter;

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
        host_results_buffer = host_result;
    }
    MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                       "host_activations",
                       host_inputs_buffer,
                       inputs_buffer_size);

    int current_row = 0;
    float* current_result = host_results_buffer;
    float* current_weights_loc = host_weights;
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

        const size_t weights_buffer_size =
                (curr_iter->cols + curr_iter->align_pad) * curr_iter->rows *
                sizeof(float);
        MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                           "host_weights",
                           current_weights_loc,
                           weights_buffer_size);
        size_t result_size = NUM_TEST_CASES * partial_layer.inputs.rows *
                             partial_layer.outputs.cols;

        inner_product_layer_hw_dispatch(host_inputs_buffer,
                                        current_weights_loc,
                                        &partial_layer,
                                        input_in_spad0,
                                        do_bias,
                                        current_result,
                                        result_size,
                                        device);

        PRINT_MSG("Partial results:\n");
        PRINT_DEBUG_V(current_result,
                      curr_layer->inputs.rows * NUM_TEST_CASES,
                      curr_iter->cols,
                      curr_iter->cols);

        current_row += curr_iter->rows - 1;
        current_result += result_size;
        current_weights_loc += iter_weights_size;
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
                is_supported_activation_func(curr_layer->type, act_func);
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
                 host_result,
                 output_cols);                 // dst buffer.
        float* biases = host_weights + (curr_layer->weights.rows - 1) *
                                               curr_layer->weights.cols;
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
            smiv_activation_function(
                    host_results_buffer, curr_layer, host_result, device);
        }
    }

    if (needs_multiple_iter) {
        free(host_inputs_buffer);
    }
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        free(host_results_buffer);
    }

    free_work_cfg(&fc_cfgs);
}

void inner_product_layer_impl_colwise(float* host_activations,
                                      float* host_weights,
                                      layer_t* curr_layer,
                                      float* host_result,
                                      device_t* device,
                                      bool input_in_spad0) {
    fc_cfg_t fc_cfgs = inner_product_divide_work_colwise(curr_layer);
    printf("Inner product layer %d work configuration:\n", curr_layer->num);
    print_work_cfg(&fc_cfgs);

    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    MAP_ARRAY(kInnerProductHw, host_activations, INPUT_BYTES(curr_layer, 0));

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
        host_results_buffer = host_result;
    }

    int current_col = 0;
    float* current_result = host_results_buffer;
    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        layer_t partial_layer = *curr_layer;
        dims_t* curr_iter = &fc_cfgs.iteration[it];
        partial_layer.weights = *curr_iter;
        partial_layer.outputs.cols = curr_iter->cols;
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

        MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                           "host_weights",
                           host_weights_buffer,
                           weights_buffer_size);

        size_t result_size = NUM_TEST_CASES * partial_layer.outputs.rows *
                             partial_layer.outputs.cols;

        inner_product_layer_hw_dispatch(host_activations,
                                        host_weights_buffer,
                                        &partial_layer,
                                        input_in_spad0,
                                        true,  // do_bias
                                        current_result,
                                        result_size,
                                        device);

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
    // final result array (host_result).
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        int output_size =
                curr_layer->outputs.cols + curr_layer->outputs.align_pad;
        ARRAY_2D(float, _host_results, host_result, output_size);  // dst buffer.
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

    free_work_cfg(&fc_cfgs);
}

void inner_product_layer_impl(float* host_activations,
                              float* host_weights,
                              layer_t* layers,
                              int lnum,
                              float* host_result,
                              device_t* device) {
    static float* current_result_loc = NULL;
    if (current_result_loc == NULL) {
        current_result_loc = g_spad1;
    } else if (current_result_loc == g_spad0) {
        current_result_loc = g_spad1;
    } else if (current_result_loc == g_spad1) {
        current_result_loc = g_spad0;
    }
    bool input_in_spad0 = (current_result_loc == g_spad1);
    float* host_weights_layer =
            host_weights + get_weights_loc_for_layer(layers, lnum);
    layer_t* curr_layer = &layers[lnum];

    PRINT_MSG("Weights:\n");
    PRINT_DEBUG(host_weights_layer,
                curr_layer->weights.rows,
                curr_layer->weights.cols,
                curr_layer->weights.cols + curr_layer->weights.align_pad);

    // Dynamically pick one to use, based on whether weights or inputs are
    // better. If inputs is bigger, we'll divide the work columnwise in the
    // weights and reorder the weights instead of the inputs, which are larger.
    // But if weights are bigger, we'll divide the work rowwise in the weights.
    // This means we can just pass a pointer to the current location in the
    // weights and only reorder the inputs (the smaller input to the GEMM).
    int input_size = get_input_activations_size(curr_layer);
    int weight_size = get_num_weights_layer(layers, lnum);
    INFO_MSG("Input size: %d, weight size: %d\n", input_size, weight_size);
    if (input_size > weight_size) {
        INFO_MSG("Running colwise inner product.\n");
        inner_product_layer_impl_colwise(host_activations,
                                         host_weights_layer,
                                         curr_layer,
                                         host_result,
                                         device,
                                         input_in_spad0);
    } else {
        INFO_MSG("Running rowwise inner product.\n");
        inner_product_layer_impl_rowwise(host_activations,
                                         host_weights_layer,
                                         curr_layer,
                                         host_result,
                                         device,
                                         input_in_spad0);
    }
}
