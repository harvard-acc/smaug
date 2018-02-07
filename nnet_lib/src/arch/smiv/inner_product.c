#include <assert.h>
#include <string.h>

#include "arch/smiv_common.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef void (*tiled_inner_product_impl)(
        float*, float*, layer_t*, float*, device_t*, bool);

void inner_product_layer_hw_impl(float* host_activations,
                                 float* host_weights,
                                 float* local_weights,
                                 float* local_inputs,
                                 float* results,
                                 layer_t* all_layers,
                                 int lnum,
                                 bool do_bias,
                                 bool use_pipelined_dma) {
    if (all_layers[lnum].weights_req == IO_DMA) {
        int weights_size = get_num_weights_layer(all_layers, lnum);
        if (!do_bias)
            weights_size -= all_layers[lnum].weights.cols;
        weights_size *= sizeof(float);
        setReadyBits(local_weights, UMEM_SIZE, 0);
        if (use_pipelined_dma) {
            divide_and_send_dma_req(host_weights, local_weights, weights_size,
                                    LOG_PAGE_SIZE, true);
        } else {
            dmaLoad(local_weights, host_weights, weights_size);
        }
    }

    if (all_layers[lnum].input_req == IO_DMA) {
        int activations_size = INPUT_BYTES(all_layers, lnum);
        setReadyBits(local_inputs, SPAD_SIZE, 0);
        if (use_pipelined_dma)
            divide_and_send_dma_req(host_activations,
                                    local_inputs,
                                    activations_size,
                                    LOG_PAGE_SIZE,
                                    true);
        else
            dmaLoad(local_inputs, host_activations, activations_size);
    }

    matrix_multiply_with_bias_smiv(
            local_inputs,
            local_weights,
            all_layers[lnum].inputs.rows * NUM_TEST_CASES,
            all_layers[lnum].weights.rows,
            all_layers[lnum].weights.cols + all_layers[lnum].weights.align_pad,
            all_layers[lnum].inputs.align_pad,
            all_layers[lnum].activation,
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
                            float* host_result,
                            bool use_pipelined_dma) {
    bool output_dma_req = (all_layers[lnum].output_req == IO_DMA);
    size_t result_size = OUTPUT_BYTES(all_layers, lnum);
    if (input_in_spad0) {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad0,
                                    spad1,
                                    all_layers,
                                    lnum,
                                    do_bias,
                                    use_pipelined_dma);
        if (output_dma_req) {
            if (use_pipelined_dma) {
                divide_and_send_dma_req(
                        host_result, spad1, result_size, LOG_PAGE_SIZE, false);
            } else {
                store_output_activations_dma(
                        host_result, spad1, &all_layers[lnum]);
            }
        }
    } else {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad1,
                                    spad0,
                                    all_layers,
                                    lnum,
                                    do_bias,
                                    use_pipelined_dma);
        if (output_dma_req) {
            if (use_pipelined_dma) {
                divide_and_send_dma_req(
                        host_result, spad0, result_size, LOG_PAGE_SIZE, false);
            } else {
                store_output_activations_dma(
                        host_result, spad0, &all_layers[lnum]);
            }
        }
    }
}

// HW accelerated inner product, using ACP for output data movement.
//
// All arguments prefixed with host_ are host memory pointers. acp_result is
// the host result pointer to be accessed over ACP.
void inner_product_layer_acp_result_hw(float* host_activations,
                                       float* host_weights,
                                       float* acp_result,
                                       float* umem,
                                       float* spad0,
                                       float* spad1,
                                       layer_t* all_layers,
                                       int lnum,
                                       bool input_in_spad0,
                                       bool do_bias,
                                       bool use_pipelined_dma) {
    if (input_in_spad0) {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad0,
                                    acp_result,
                                    all_layers,
                                    lnum,
                                    do_bias,
                                    use_pipelined_dma);
    } else {
        inner_product_layer_hw_impl(host_activations,
                                    host_weights,
                                    umem,
                                    spad1,
                                    acp_result,
                                    all_layers,
                                    lnum,
                                    do_bias,
                                    use_pipelined_dma);
    }
}

// HW accelerated inner product, using ACP for everything (inputs, weights and
// result).
void inner_product_layer_acp_hw(float* acp_activations,
                                float* acp_weights,
                                float* acp_result,
                                layer_t* all_layers,
                                int lnum,
                                bool do_bias) {
    inner_product_layer_hw_impl(NULL,
                                NULL,
                                acp_weights,
                                acp_activations,
                                acp_result,
                                all_layers,
                                lnum,
                                do_bias,
                                false);
}

// HW accelerated inner product, using harware cache for everything (inputs,
// weights and result).
void inner_product_layer_cache_hw(float* cache_activations,
                                  float* cache_weights,
                                  float* cache_result,
                                  layer_t* all_layers,
                                  int lnum,
                                  bool do_bias) {
    inner_product_layer_hw_impl(NULL,
                                NULL,
                                cache_weights,
                                cache_activations,
                                cache_result,
                                all_layers,
                                lnum,
                                do_bias,
                                false);
}

// Decompress a CSR array in HW.
//
// The compressed data will be sent to one of the scratchpads, and the
// decompressed data will be written to the UMEM.
//
// This one function can be used for any of the available input mechanisms
// (dma/acp/cache), although much of the complexity is a result of needing to
// tile the CSR array to fit in the available sceratchpad space. However, the
// output will always be placed into the UMEM.
//
// Arguments:
//   dma_weights: The compressed data, accessed via DMA.
//   acp_weights: The compressed data, accessed via ACP.
//   cache_weights: The compressed data, accessed via HW cache.
//   cmp_col_offset: The offset (32-bit granularity) into the source compressed
//       data at which the column indices start.
//   cmp_row_offset: The offset (32-bit granularity) into the source compressed
//       data at which the row indices start.
//   dest_offset: The offset (32-bit granularity) into the destination buffer
//       from where the data should start getting written. This is required to
//       support tiled decompression.
//   compressed_size: The size (bytes) of the complete source CSR array.
//   decompressed_size: The size (bytes) that the array will take up once
//       decompressed.
//   input_in_spad0: Send the CSR data to spad0 if true.
//   copy_mechanism: Which mechanism to use for sending the input.
//   spad0: SPAD0 pointer.
//   spad1: SPAD1 pointer.
//   umem: UMEM pointer.
void decompress_packed_csr_smiv_hw(uint32_t* dma_weights,
                                   uint32_t* acp_weights,
                                   uint32_t* cache_weights,
                                   int cmp_col_offset,
                                   int cmp_row_offset,
                                   int dest_offset,
                                   dims_t* data_dims,
                                   size_t compressed_size,
                                   size_t decompressed_size,
                                   bool input_in_spad0,
                                   io_req_t copy_mechanism,
                                   float* spad0,
                                   float* spad1,
                                   float* umem) {
    PRINT_MSG("Decompressing CSR data!\n");
    ASSERT(compressed_size <= SPAD_SIZE &&
           "CSR array size exceeds scratchpad capacity!");
    // The umem must be zeroed first.
    int num_rows = decompressed_size / (VECTOR_SIZE * sizeof(float));
    int start_row = dest_offset / VECTOR_SIZE;
    VEC_ARRAY_1D(v8fp_t, _umem, umem);
    decompress_reset:
    for (int i = start_row; i < start_row + num_rows; i++)
        _umem[i] = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };

    if (copy_mechanism == IO_DMA) {
        if (input_in_spad0) {
            setReadyBits(spad0, compressed_size, 0);
            dmaLoad(spad0, dma_weights, compressed_size);
            decompress_packed_csr_data_smiv_fxp(
                    (uint32_t*)spad0, cmp_col_offset, cmp_row_offset,
                    dest_offset, data_dims, umem);
        } else {
            setReadyBits(spad1, compressed_size, 0);
            dmaLoad(spad1, dma_weights, compressed_size);
            decompress_packed_csr_data_smiv_fxp(
                    (uint32_t*)spad1, cmp_col_offset, cmp_row_offset,
                    dest_offset, data_dims, umem);
        }
    } else if (copy_mechanism == IO_ACP) {
        decompress_packed_csr_data_smiv_fxp(acp_weights, cmp_col_offset,
                                            cmp_row_offset, dest_offset,
                                            data_dims, umem);
    } else if (copy_mechanism == IO_CACHE) {
        decompress_packed_csr_data_smiv_fxp(cache_weights, cmp_col_offset,
                                            cmp_row_offset, dest_offset,
                                            data_dims, umem);
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
    bool use_pipelined_dma = device->use_pipelined_dma;
    io_req_t input_req = layer->input_req;
    io_req_t weights_req = layer->weights_req;
    io_req_t output_req = layer->output_req;
    if (output_req != IO_NONE) {
        const char* results_var_name =
                output_req == IO_DMA ? "host_result"
                                     : output_req == IO_ACP ? "acp_result"
                                                            : "cache_result";
        MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                           results_var_name,
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
        if (output_req == IO_ACP) {
            // Use ACP for results.
            INVOKE_KERNEL_PROF(kInnerProductHw,
                               layer->num,
                               inner_product_layer_acp_result_hw,
                               activations,
                               weights,
                               results,
                               g_umem,
                               g_spad0,
                               g_spad1,
                               layer,
                               0,
                               input_in_spad0,
                               do_bias,
                               use_pipelined_dma);
        } else {
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
                               results,
                               use_pipelined_dma);
        }
    } else if (input_req == IO_ACP) {
        INVOKE_KERNEL_PROF(kInnerProductHw,
                           layer->num,
                           inner_product_layer_acp_hw,
                           activations,
                           weights,
                           results,
                           layer,
                           0,
                           do_bias);
    } else if (input_req == IO_CACHE) {
        INVOKE_KERNEL_PROF(kInnerProductHw,
                           layer->num,
                           inner_product_layer_cache_hw,
                           activations,
                           weights,
                           results,
                           layer,
                           0,
                           do_bias);
    }
}

void inner_product_layer_impl_rowwise(float* host_activations,
                                      float* host_weights,
                                      layer_t* curr_layer,
                                      float* host_result,
                                      device_t* device,
                                      bool input_in_spad0) {
    INFO_MSG("Running rowwise inner product.\n");
    fc_cfg_t fc_cfgs = inner_product_divide_work_rowwise(curr_layer);
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_work_cfg(&fc_cfgs);
    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    bool do_bias_in_software =
            fc_cfgs.iteration[fc_cfgs.num_iterations - 1].rows == 1;

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

    io_req_t input_req = curr_layer->input_req;
    const char* activations_var_name = input_req == IO_DMA
            ? "host_activations"
            : input_req == IO_ACP ? "acp_activations" : "cache_activations";
    MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                       activations_var_name,
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
        activation_type act_func = curr_layer->activation;
        bool do_hw_activation =
                device->use_hw_activation_func &&
                is_supported_activation_func(curr_layer->type, act_func);
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
        if (curr_layer->wgt_storage_type == PackedCSR) {
            // If this is not the last iteration, then pre-emptively subtract
            // one from the rows to get rid of decompressing an extra row for
            // nothing.
            layer_t temp_layer = partial_layer;
            if (!is_last_iter)
                temp_layer.weights.rows--;
            packed_csr_array_t* src_csr =
                    (packed_csr_array_t*)temp_layer.host_weights_buffer;
            csr_tile_list* tile_list = tile_packed_csr_array_t(
                    src_csr, &temp_layer.weights, current_row, SPAD_SIZE);
            assert(tile_list->len > 0 && "CSR tile list cannot be empty!");
            csr_tile* curr_tile = tile_list->head;
            int dest_offset = 0;
            do {
                packed_csr_array_t* array = curr_tile->array;
                dims_t dims = (dims_t){ curr_tile->num_rows,
                                        temp_layer.weights.cols,
                                        temp_layer.weights.height,
                                        temp_layer.weights.align_pad };
                MAP_ARRAY_TO_ACCEL(kInnerProductHw, "dma_weights",
                                   array->vals,
                                   array->total_buf_size);
                INVOKE_KERNEL_PROF(kInnerProductHw,
                                   curr_layer->num,
                                   decompress_packed_csr_smiv_hw,
                                   array->vals,  // DMA
                                   array->vals,  // ACP
                                   array->vals,  // Cache
                                   array->col_idx - array->vals,
                                   array->row_idx - array->vals,
                                   dest_offset,
                                   &dims,
                                   array->total_buf_size,
                                   curr_tile->eff_total_bytes,
                                   !input_in_spad0,  // Don't overwrite inputs!
                                   device->cpu_default_offload,
                                   g_spad0,
                                   g_spad1,
                                   g_umem);
                dest_offset += (curr_tile->eff_total_bytes / sizeof(uint32_t));
                curr_tile = curr_tile->next_tile;
            } while (curr_tile);
            // Now that we've decompressed the weights, we don't need to DMA
            // them again.
            free_csr_tile_list(tile_list);
            partial_layer.weights_req = IO_NONE;
            PRINT_MSG("Weights:\n");
            PRINT_DEBUG(g_umem,
                        partial_layer.weights.rows,
                        partial_layer.weights.cols,
                        partial_layer.weights.cols +
                                partial_layer.weights.align_pad);
        }

        io_req_t weights_req = partial_layer.weights_req;
        const size_t weights_buffer_size =
                (curr_iter->cols + curr_iter->align_pad) * curr_iter->rows *
                sizeof(float);
        if (weights_req != IO_NONE) {
            const char* weights_var_name = weights_req == IO_DMA
                                                   ? "host_weights"
                                                   : weights_req == IO_ACP
                                                             ? "acp_weights"
                                                             : "cache_weights";
            MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                               weights_var_name,
                               curr_dense_weights_loc,
                               weights_buffer_size);
        }
        size_t result_size = NUM_TEST_CASES * partial_layer.inputs.rows *
                             partial_layer.outputs.cols;
        inner_product_layer_hw_dispatch(host_inputs_buffer,
                                        curr_dense_weights_loc,
                                        &partial_layer,
                                        input_in_spad0,
                                        do_bias,
                                        current_result,
                                        result_size,
                                        device);

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
    INFO_MSG("Running colwise inner product.\n");
    INFO_MSG("Inner product layer %d work configuration:\n", curr_layer->num);
    print_work_cfg(&fc_cfgs);

    bool needs_multiple_iter = (fc_cfgs.num_iterations > 1);
    io_req_t input_req = curr_layer->input_req;
    const char* activations_var_name = input_req == IO_DMA
            ? "host_activations"
            : input_req == IO_ACP ? "acp_activations" : "cache_activations";
    MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                       activations_var_name,
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
        host_results_buffer = host_result;
    }

    int current_col = 0;
    float* current_result = host_results_buffer;
    for (unsigned it = 0; it < fc_cfgs.num_iterations; it++) {
        layer_t partial_layer = *curr_layer;
        dims_t* curr_iter = &fc_cfgs.iteration[it];
        partial_layer.weights = *curr_iter;
        partial_layer.outputs.cols = curr_iter->cols;

        activation_type act_func = curr_layer->activation;
        bool do_hw_activation =
                device->use_hw_activation_func &&
                is_supported_activation_func(curr_layer->type, act_func);
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

        if (input_req != IO_NONE) {
            const char* weights_var_name =
                    input_req == IO_DMA ? "host_weights"
                                        : input_req == IO_ACP ? "acp_weights"
                                                              : "cache_weights";
            MAP_ARRAY_TO_ACCEL(kInnerProductHw,
                               weights_var_name,
                               host_weights_buffer,
                               weights_buffer_size);
        }

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
    layer_t* curr_layer = &layers[lnum];
    float* host_weights_layer = (float*)curr_layer->host_weights_buffer;

    if (curr_layer->wgt_storage_type == Uncompressed) {
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
                (input_size > weight_size) ? &inner_product_layer_impl_colwise
                                           : &inner_product_layer_impl_rowwise;
        impl(host_activations, host_weights_layer, curr_layer, host_result,
             device, input_in_spad0);
    } else if (curr_layer->wgt_storage_type == PackedCSR) {
        // If the weights are stored in CSR format, we can only do row-wise
        // tiling.
        INFO_MSG("Running rowwise inner product for packed CSR weights.\n");
        inner_product_layer_impl_rowwise(host_activations,
                                         host_weights_layer,
                                         curr_layer,
                                         host_result,
                                         device,
                                         input_in_spad0);
    } else if (curr_layer->wgt_storage_type == CSR) {
        fprintf(stderr, "Inner product layer for unpacked CSR weights is not "
                        "supported!\n");
        exit(1);
    }

}
