#include <assert.h>
#include <math.h>
#include <string.h>

#include "nnet_fwd.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/smiv.h"
#include "core/zeropad.h"
#include "utility/utility.h"
#include "arch/common.h"
#include "arch/interface.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == SMIV

// Each SMIV block has two scratchpads of 64KB each, but the real accelerator
// operates on 16-bit data, whereas we are using 32-bit data. To make sure we
// can fit the same size inputs, we double the per-scratchpad size.
#define SPAD_SIZE 131072

// The UMEM on the NIC is 3 blocks of 1MB each.
#define UMEM_SIZE (3*1048576)

typedef struct _conv_cfg_t {
    // An array of dim_t objects. Specify the rows, cols, channels, and padding
    // for each iteration.
    dims_t* iteration;
    // Size of the array.
    unsigned num_iterations;
} conv_cfg_t;

// Use the same accelerator id for both the convolutional and FC blocks. This
// means we will simulate only ONE datapath instead of two, which means that
// the two blocks can share the scratchpads (without any more infrastructure
// changes). The key is that we still trace the functions at the _hw level, so
// that Aladdin will exit after simulating each block, and we can return
// control to the CPU at the right places.  In contrast, if we used two
// different ids, we would have two different datapaths that could not share
// data directly.
unsigned kConvolutionHw = 0x0003;
unsigned kInnerProductHw = 0x0003;
unsigned kReductionHw = 0x0003;

void inner_product_layer_hw(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result) {
    bool run_activation = layers[lnum].activation != NONE;
    grab_matrix_dma(weights, lnum, layers);
    if (layers[lnum].needs_input_dma_load)
        grab_input_activations_dma(activations, lnum, layers);
    matrix_multiply_with_bias_smiv(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            layers[lnum].inputs.align_pad, run_activation, result);
    if (layers[lnum].needs_output_dma_store)
        store_output_activations_dma(result, lnum, layers);
}

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result) {
    MAP_ARRAY(kInnerProductHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, weights, OUTPUT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, result, WEIGHT_BYTES(layers, lnum));
    INVOKE_KERNEL(kInnerProductHw, inner_product_layer_hw, activations, weights,
                  layers, lnum, result);
    return result;
}

void reduction_hw(float* unreduced_activations,
                  layer_t curr_layer,
                  float* result) {
    reduction_smiv(unreduced_activations, curr_layer, result);
    if (curr_layer.needs_output_dma_store) {
        int size = curr_layer.outputs.rows *
                   (curr_layer.outputs.cols + curr_layer.outputs.align_pad) *
                   curr_layer.outputs.height;
        dmaStore(result, 0, 0, size);
    }
}

void convolution_layer_hw(float* activations,
                          float* weights,
                          layer_t curr_layer,
                          float* result) {
    int kernel_size = curr_layer.weights.rows *
                      (curr_layer.weights.cols + curr_layer.weights.align_pad) *
                      curr_layer.weights.height;
    dmaLoad(weights, 0, 0, kernel_size);
    if (curr_layer.needs_input_dma_load) {
        int size = curr_layer.inputs.rows *
                   (curr_layer.inputs.cols + curr_layer.inputs.align_pad) *
                   curr_layer.inputs.height;
        dmaLoad(activations, 0, 0, size);
    }

    convolution3d_smiv(activations, weights, curr_layer, result);
}

// Find a good way to pack the convolution into the accelerator.
conv_cfg_t convolution_divide_work(layer_t* layers, int lnum) {
    conv_cfg_t conv_cfgs;
    unsigned total_input_bytes = INPUT_BYTES(layers, lnum) / NUM_TEST_CASES;
    // This is the unreduced output for a single output channel.
    unsigned total_output_bytes =
            layers[lnum].outputs.rows *
            (layers[lnum].outputs.cols + layers[lnum].outputs.align_pad) *
            layers[lnum].inputs.height * sizeof(float);
    if (total_input_bytes > UMEM_SIZE) {
        printf("A single input image exceeds the capacity of the UMEM, which "
               "is not supported!\n");
        assert(false);
    }
    if (total_output_bytes <= SPAD_SIZE) {
        PRINT_MSG("Entire input problem fits into the local memory.\n");
        conv_cfgs.iteration = (dims_t*)malloc(sizeof(dims_t));
        conv_cfgs.iteration[0].rows = layers[lnum].inputs.rows;
        conv_cfgs.iteration[0].cols = layers[lnum].inputs.cols;
        conv_cfgs.iteration[0].height = layers[lnum].inputs.height;
        conv_cfgs.iteration[0].align_pad =
                calc_padding(conv_cfgs.iteration[0].cols, DATA_ALIGNMENT);
        conv_cfgs.num_iterations = 1;
        return conv_cfgs;
    }

    // Divide the problem up per input channel.

    unsigned output_channel_size =
            layers[lnum].outputs.rows *
            (layers[lnum].outputs.cols + layers[lnum].outputs.align_pad) *
            sizeof(float);
    unsigned input_channels = layers[lnum].inputs.height;

    unsigned max_channels_per_iter = SPAD_SIZE / output_channel_size;
    if (max_channels_per_iter >= 2) {
        PRINT_MSG("We can fit at least 2 unreduced input channels at once.\n");
        conv_cfgs.num_iterations =
                ceil((float)input_channels / max_channels_per_iter);
        conv_cfgs.iteration =
                (dims_t*)malloc(conv_cfgs.num_iterations * sizeof(dims_t));
        unsigned total_channels = input_channels;
        for (int i = 0; i < conv_cfgs.num_iterations; i++) {
            conv_cfgs.iteration[i].rows = layers[lnum].inputs.rows;
            conv_cfgs.iteration[i].cols = layers[lnum].inputs.cols;
            conv_cfgs.iteration[i].height =
                    min(total_channels, max_channels_per_iter);
            conv_cfgs.iteration[i].align_pad =
                    calc_padding(conv_cfgs.iteration[i].cols, DATA_ALIGNMENT);
            total_channels -= max_channels_per_iter;
        }
        return conv_cfgs;
    }

    // We can't fit more than a single channel onto the accelerator, which
    // means we won't be able to reduce in the accelerator. So now we have to
    // start chopping up the image into blocks.

    assert(false && "Tiled input handling is not yet supported!\n");
    return conv_cfgs;
}

void convolution_runner(float* host_activations,
                        float* host_weights,
                        layer_t* layers,
                        int lnum,
                        float* host_result) {

    layer_t curr_layer = layers[lnum];
    const int input_height = curr_layer.inputs.height;
    const int input_rows= curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = curr_layer.inputs.align_pad;
    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    const int num_kerns = curr_layer.outputs.height;
    const int result_2d_size = result_height * (result_width + result_pad);

    ARRAY_4D(float, _a, host_activations, input_height, input_rows,
             input_cols + input_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);
    ARRAY_4D(float, _result, host_result, num_kerns, result_height,
             result_width + result_pad);

    float* umem = (float*)malloc(UMEM_SIZE);
    float* spad0 = (float*)malloc(SPAD_SIZE);
    float* spad1 = (float*)malloc(SPAD_SIZE);

    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "activations", umem, UMEM_SIZE);
    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "weights", spad0, SPAD_SIZE);
    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "result", spad1, SPAD_SIZE);
    MAP_ARRAY_TO_ACCEL(kReductionHw, "unreduced_activations", spad1, SPAD_SIZE);

    conv_cfg_t conv_cfgs = convolution_divide_work(layers, lnum);
    // temp_result stores the partially reduced results of each iteration.
    float* temp_result = (float*)malloc(
            result_2d_size * conv_cfgs.num_iterations * sizeof(float));

    // TODO: Fix this - we shouldn't use this function outside of an
    // accelerated kernel.
    grab_matrix_dma(host_weights, lnum, layers);
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        for (int kern = 0; kern < num_kerns; kern++) {
            PRINT_MSG("Kernel %d\n", kern);
            PRINT_DEBUG4D(&_kernels[kern][0][0][0],
                          k_width,
                          k_width + k_pad,
                          input_height);
            unsigned start_chan = 0;
            float* result_loc = temp_result;
            for (int iter = 0; iter < conv_cfgs.num_iterations; iter++) {
                PRINT_MSG("Iteration %d\n", iter);
                dims_t iter_cfg = conv_cfgs.iteration[iter];
                // Fake the DMA until I get the chance to do DMA v3.
                unsigned input_size = iter_cfg.rows *
                                      (iter_cfg.cols + iter_cfg.align_pad) *
                                      iter_cfg.height;
                unsigned kernel_size = curr_layer.weights.rows *
                                       (curr_layer.weights.cols +
                                        curr_layer.weights.align_pad) *
                                       iter_cfg.height;
                // TODO: Eliminate this extra memcpy once we have proper DMA
                // to/from arbitrary memory addresses working.
                memcpy(umem, &_a[img][start_chan][0][0],
                       input_size * sizeof(float));
                memcpy(spad0, &_kernels[kern][start_chan][0][0],
                       kernel_size * sizeof(float));

                // Create a new layer description for this iteration.
                layer_t partial_layer = curr_layer;
                partial_layer.inputs.height = iter_cfg.height;
                partial_layer.outputs.height = iter_cfg.height;
                partial_layer.activation = conv_cfgs.num_iterations == 1
                                                   ? curr_layer.activation
                                                   : NONE;
                INVOKE_KERNEL(kConvolutionHw, convolution_layer_hw, umem, spad0,
                              partial_layer, spad1);

                // Reduce the results.
                INVOKE_KERNEL(kReductionHw, reduction_hw, spad1, partial_layer,
                              umem);

                // Copy the partially reduced results to result.
                // TODO: Eliminate this extra step once we have proper DMA
                // to/from arbitrary memory addresses working...or just put it
                // into the UMEM.
                memcpy(result_loc, umem, result_2d_size * sizeof(float));
                result_loc += result_2d_size;
                start_chan += iter_cfg.height;
            }

            // Finish off the reduction here.
            if (conv_cfgs.num_iterations > 1) {
                result_loc = temp_result;

                int result_iter =
                        ceil(result_2d_size * conv_cfgs.num_iterations /
                             (float)SPAD_SIZE);
                assert(result_iter == 1 &&
                       "Only support 1 last iteration of reduction!");
                int num_result_chans = min(
                        conv_cfgs.num_iterations, SPAD_SIZE / result_2d_size);

                // Create a new layer description for this iteration.
                layer_t partial_layer = curr_layer;
                partial_layer.inputs.height = num_result_chans;
                partial_layer.outputs.height = 1;
                for (int iter = 0; iter < result_iter; iter++) {
                    memcpy(spad1, result_loc,
                           result_2d_size * num_result_chans * sizeof(float));
                    PRINT_MSG("Final reduction round %d\n", iter);
                    INVOKE_KERNEL(kReductionHw, reduction_hw, spad1,
                                  partial_layer, umem);
                    memcpy(result_loc, umem, result_2d_size * sizeof(float));
                    result_loc += result_2d_size;
                }
            }
            memcpy(&_result[img][kern][0][0], temp_result,
                   result_2d_size * sizeof(float));
        }
    }
    free(conv_cfgs.iteration);
    free(spad0);
    free(spad1);
    free(umem);
    free(temp_result);
}

result_buf convolution_layer(float* activations,
                             float* weights,
                             layer_t* layers,
                             int lnum,
                             float* result) {

    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        // TODO: Replace this with a memcpy implementation.
        copy_zeropad(activations, curr_layer, result);
        PRINT_MSG("After zeropadding:\n");
        PRINT_DEBUG4D(result,
                      curr_layer.inputs.rows,
                      curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                      curr_layer.inputs.height);
        PRINT_DEBUG4D(weights, curr_layer.weights.rows,
                      curr_layer.weights.cols + curr_layer.weights.align_pad,
                      curr_layer.weights.height);
        convolution_runner(result, weights, layers, lnum, activations);

        return activations;
    }
    convolution_runner(activations, weights, layers, lnum, result);
    return result;
}

// Software implementation. SMIV doesn't accelerate pooling.
result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.pool == MAX) {
        max_pooling(activations, result, layers[lnum]);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    return result;
}

result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     float* sigmoid_table) {
    result_buf result_loc = run_layer_skip_activation_func(
            activations, weights, layers, layer_num, result, sigmoid_table);

    // Activation functions are handled as part of the matrix multiply /
    // convolution, rather than being treated as a separate block.
    return result_loc;
}

// Set the dmaLoad/dmaStore required flags for each layer.
//
// Since SMIV can share scratchpads between the conv/fc blocks, we only need
// DMA if we need to send data back to the CPU.
void set_dma_requirements(network_t* network) {
    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        // The input layer is easy.
        if (layer_num == 0) {
            network->layers[layer_num].needs_input_dma_load = false;
            network->layers[layer_num].needs_output_dma_store = true;
            continue;
        }
        // First, determine if we need to dma store the output.
        if (layer_num == network->depth - 1 ||
            network->layers[layer_num].activation == SIGMOID ||
            network->layers[layer_num].type == POOLING ||
            network->layers[layer_num].input_preprocessing == FLATTEN ||
            network->layers[layer_num + 1].type == POOLING ||
            network->layers[layer_num + 1].type == SOFTMAX) {
            network->layers[layer_num].needs_output_dma_store = true;
        } else {
            network->layers[layer_num].needs_output_dma_store = false;
        }

        // Whether we need to load the input on this layer is just whether we
        // had to store the outputs in the previous layer.
        network->layers[layer_num].needs_input_dma_load =
                network->layers[layer_num - 1].needs_output_dma_store;
    }

    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        printf("Layer %d: dmaLoad = %d, dmaStore = %d\n", layer_num,
               network->layers[layer_num].needs_input_dma_load,
               network->layers[layer_num].needs_output_dma_store);
    }

}

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, activations and result.
void nnet_fwd(farray_t activations,
              farray_t weights,
              farray_t result,
              network_t network,
              float* sigmoid_table) {

    int l;
    layer_t curr_layer;

    // Alternate between reading from/writing to activations and result so we
    // can avoid copying matrices. The initial activations is obviously in
    // "activations", so that's where we start.
    result_buf result_loc = activations.d;

    if (PRINT_DATA_AND_WEIGHTS) {
        print_data_and_weights(activations.d, weights.d, network.layers[0]);
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;

    set_dma_requirements(&network);

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 0; l < network.depth; l++) {
        curr_layer = network.layers[l];

        if (result_loc == result.d) {
            result_loc = run_layer(result.d, weights.d, network.layers, l,
                                   activations.d, sigmoid_table);
        } else {
            result_loc = run_layer(activations.d, weights.d, network.layers, l,
                                   result.d, sigmoid_table);
        }
    }

    network.layers[network.depth - 1].result_in_temp = (result_loc == result.d);

    if (result_loc == result.d)
        dmaStore(result.d, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(activations.d, 0, 0,
                 NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(network.layers, 0, 0, network.depth * sizeof(layer_t));
}

#endif
