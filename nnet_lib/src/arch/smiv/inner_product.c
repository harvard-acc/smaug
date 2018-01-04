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
                                 int lnum) {
    activation_type act_func = all_layers[lnum].activation;
    bool run_activation = act_func == RELU || act_func == RELU_THRESHOLD;
    int weights_size = get_num_weights_layer(all_layers, lnum) * sizeof(float);
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
                            float* host_result) {
    bool output_dma_req = (all_layers[lnum].output_req == IO_DMA);
    if (input_in_spad0) {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad0,
                                    spad1,
                                    all_layers,
                                    lnum);
        if (output_dma_req)
            store_output_activations_dma(host_result, spad1, &all_layers[lnum]);
    } else {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad1,
                                    spad0,
                                    all_layers,
                                    lnum);
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
                                bool input_in_spad0) {
    if (input_in_spad0) {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad0,
                                    acp_result,
                                    all_layers,
                                    lnum);
    } else {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad1,
                                    acp_result,
                                    all_layers,
                                    lnum);
    }
}

// Returns true if this inner product layer will require multiple iterations.
bool inner_product_needs_work_division(layer_t* layers, int lnum) {
    const unsigned total_weight_bytes = WEIGHT_BYTES(layers, lnum);
    return total_weight_bytes > UMEM_SIZE;
}

// Divides the work for a FC layer into several iterations on SMIV.
//
// Work division is required when the number of weights exceeds what can be fit
// on the UMEM. The weight matrix is In x On, where In is the number of input
// neurons and On is the number of output neurons. Dividing the work means to
// do the matrix multiply in groups of In x W, where W = On/iterations.
fc_cfg_t inner_product_divide_work(layer_t* layers, int lnum) {
    fc_cfg_t fc_cfgs;
    // TODO: These are not quite the right constraints.
    const unsigned total_input_bytes =
            INPUT_BYTES(layers, lnum) / NUM_TEST_CASES;
    if (total_input_bytes > SPAD_SIZE) {
        printf("A single input does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    const unsigned total_output_bytes =
            OUTPUT_BYTES(layers, lnum) / NUM_TEST_CASES;
    if (total_output_bytes > SPAD_SIZE) {
        printf("A single output does not fit in the SPAD, which is not "
               "supported!\n");
        assert(false);
    }
    if (!inner_product_needs_work_division(layers, lnum)) {
        // No work division means to return an fc_cfg_t that is holds the
        // entire weights.
        init_work_cfg(&fc_cfgs, 1);
        fc_cfgs.iteration[0] = layers[lnum].weights;
        return fc_cfgs;
    }
    // Divide up the weights. The minimum required work (for now) is an Nx8
    // strip of weights, where N is the number of hidden neurons.
    const int num_inputs = layers[lnum].weights.rows;
    const unsigned num_neurons =
            layers[lnum].weights.cols + layers[lnum].weights.align_pad;
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
    const unsigned total_weight_bytes = WEIGHT_BYTES(layers, lnum);
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

// Copy the weights section from (0, start_col) to (num_rows, start_col +
// num_cols). This will include the biases for that section.
//
// TODO: Can we avoid this copy by using a really large align_pad?
void copy_weights_block(float* host_weights,
                        layer_t* layers,
                        int lnum,
                        int start_col,
                        int num_cols,
                        float* weights_buffer) {
    int num_rows = layers[lnum].weights.rows;
    int num_total_cols =
            layers[lnum].weights.cols + layers[lnum].weights.align_pad;
    ARRAY_2D(float, _weights, host_weights, num_total_cols);
    for (int r = 0; r < num_rows; r++) {
        memcpy(weights_buffer + r * num_cols,
               &_weights[r][start_col],
               num_cols * sizeof(float));
    }
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
    float* host_weights_layer =
            host_weights + get_weights_loc_for_layer(layers, lnum);

    PRINT_MSG("Weights:\n");
    PRINT_DEBUG(host_weights_layer,
                layers[lnum].weights.rows,
                layers[lnum].weights.cols,
                layers[lnum].weights.cols + layers[lnum].weights.align_pad);

    fc_cfg_t fc_cfgs = inner_product_divide_work(layers, lnum);
    printf("Inner product layer %d work configuration:\n", lnum);
    print_work_cfg(&fc_cfgs);

    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    MAP_ARRAY(kInnerProductHw, host_activations, INPUT_BYTES(layers, lnum));

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
        host_weights_buffer = host_weights_layer;
    }
    if (NUM_TEST_CASES > 1 && needs_multiple_iter) {
        host_results_buffer =
                (float*)malloc_aligned(OUTPUT_BYTES(layers, lnum));
    } else {
        host_results_buffer = host_result;
    }

    int current_col = 0;
    float* current_result = host_results_buffer;
    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        layer_t partial_layer = layers[lnum];
        partial_layer.weights = fc_cfgs.iteration[it];
        partial_layer.outputs.cols = fc_cfgs.iteration[it].cols;
        PRINT_MSG("FC iteration %d: weights %dx%d\n",
                    it,
                    partial_layer.weights.rows,
                    partial_layer.weights.cols);

        if (needs_multiple_iter) {
            copy_weights_block(host_weights_layer,
                               layers,
                               lnum,
                               current_col,
                               fc_cfgs.iteration[it].cols +
                                       fc_cfgs.iteration[it].align_pad,
                               host_weights_buffer);

            PRINT_DEBUG_V(host_weights_buffer,
                          fc_cfgs.iteration[it].rows,
                          fc_cfgs.iteration[it].cols,
                          fc_cfgs.iteration[it].cols +
                                  fc_cfgs.iteration[it].align_pad);
        }

        MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                           "host_weights",
                           host_weights_buffer,
                           weights_buffer_size);

        size_t result_size = NUM_TEST_CASES * partial_layer.outputs.rows *
                             partial_layer.outputs.cols;

        // If the result is to be in g_spad1, then the input is in g_spad0.
        bool input_in_spad0 = (current_result_loc == g_spad1);
        bool use_acp_offload = (device->cpu_activation_func_offload == IO_ACP);
        if (use_acp_offload) {
            MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                               "acp_result",
                               current_result,
                               result_size * sizeof(float));
            INVOKE_KERNEL_PROF(kInnerProductHw,
                               inner_product_layer_acp_hw,
                               host_activations,
                               host_weights_buffer,
                               current_result,
                               g_umem,
                               g_spad0,
                               g_spad1,
                               &partial_layer,
                               0,
                               input_in_spad0);
        } else {
            MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                               "host_result",
                               current_result,
                               result_size * sizeof(float));
            INVOKE_KERNEL_PROF(kInnerProductHw,
                               inner_product_layer_hw,
                               host_activations,
                               host_weights_buffer,
                               g_umem,
                               g_spad0,
                               g_spad1,
                               &partial_layer,
                               0,
                               input_in_spad0,
                               current_result);
        }
        PRINT_MSG_V("Partial results:\n");
        PRINT_DEBUG_V(
                current_result,
                NUM_TEST_CASES,
                fc_cfgs.iteration[it].cols,
                fc_cfgs.iteration[it].cols + fc_cfgs.iteration[it].align_pad);

        current_col += fc_cfgs.iteration[it].cols;
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
                layers[lnum].outputs.cols + layers[lnum].outputs.align_pad;
        ARRAY_2D(float, _host_results, host_result, output_size);  // dst buffer.
        current_result = host_results_buffer;  // temporary buffer.
        int curr_col = 0;
        for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
            int it_output_size = (fc_cfgs.iteration[it].cols +
                                  fc_cfgs.iteration[it].align_pad);
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
