#include <assert.h>
#include <string.h>

#include "arch/smiv_common.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "core/smiv/smiv.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == SMIV

void reduction_hw_impl(float* spad0,
                       float* spad1,
                       float* local_result,
                       bool input_in_spad0,
                       bool needs_input_load,
                       layer_t partial_layer,
                       size_t result_size,
                       float* host_result) {
    if (needs_input_load) {
        size_t input_bytes =
                result_size * partial_layer.inputs.height * sizeof(float);
        if (input_in_spad0) {
            setReadyBits(spad0, input_bytes, 0);
            dmaLoad(spad0, host_result, input_bytes);
        } else {
            setReadyBits(spad1, input_bytes, 0);
            dmaLoad(spad1, host_result, input_bytes);
        }
    }

    if (input_in_spad0)
        reduction_smiv(spad0, partial_layer, local_result);
    else
        reduction_smiv(spad1, partial_layer, local_result);
}

// Default Reduction module.
//
// Call this when we want to use DMA exclusively for moving data.
void reduction_hw(float* spad0,
                  float* spad1,
                  float* umem,
                  bool input_in_spad0,
                  bool needs_input_load,
                  layer_t partial_layer,
                  size_t result_size,
                  float* host_result) {
    reduction_hw_impl(spad0, spad1, umem, input_in_spad0, needs_input_load,
                      partial_layer, result_size, host_result);

    if (partial_layer.output_req == IO_DMA) {
        dmaStore(host_result, umem, result_size * sizeof(float));
    }
}

// ACP reduction module.
//
// The two scratchpads must remain named spad0 and spad1, but we use a
// different name for the result, called acp_result (instead of umem), to
// distinguish that the Aladdin config file should mark this array with acp.
//
// Importantly, acp_result is a pointer that corresponds to the host, unlike
// spad0/spad1/umem, which are accelerator-local memory.
void reduction_acp_hw(float* spad0,
                      float* spad1,
                      float* acp_result,
                      bool input_in_spad0,
                      bool needs_input_load,
                      layer_t partial_layer,
                      size_t result_size) {
    reduction_hw_impl(spad0, spad1, acp_result, input_in_spad0,
                      needs_input_load, partial_layer, result_size, acp_result);
}

static void convolution_layer_hw(float* host_activations,
                                 float* host_weights,
                                 float* host_results,
                                 float* umem,
                                 float* spad0,
                                 float* spad1,
                                 bool input_in_spad0,
                                 layer_t* all_layers,
                                 layer_t partial_layer,
                                 int layer_num,
                                 int img,
                                 int kern,
                                 int start_chan) {
    layer_t curr_layer = all_layers[layer_num];
    const int input_height = curr_layer.inputs.height;
    const int input_rows= curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = curr_layer.inputs.align_pad;
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;

    ARRAY_4D(float, _a, host_activations, input_height, input_rows,
             input_cols + input_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);

    // We should only DMA part of the weights.
    size_t num_weights =
            partial_layer.weights.rows * partial_layer.weights.height *
            (partial_layer.weights.cols + partial_layer.weights.align_pad);
    if (input_in_spad0) {
        setReadyBits(spad0, num_weights * sizeof(float), 0);
        dmaLoad(spad0, &_kernels[kern][start_chan][0][0],
                num_weights * sizeof(float));
    } else {
        setReadyBits(spad1, num_weights * sizeof(float), 0);
        dmaLoad(spad1, &_kernels[kern][start_chan][0][0],
                num_weights * sizeof(float));
    }
    if (partial_layer.input_req == IO_DMA) {
        size_t num_input_pixels =
                partial_layer.inputs.rows * partial_layer.inputs.height *
                (partial_layer.inputs.cols + partial_layer.inputs.align_pad);
        setReadyBits(umem, num_input_pixels * sizeof(float), 0);
        dmaLoad(umem, &_a[img][start_chan][0][0],
                num_input_pixels * sizeof(float));
    }

    if (input_in_spad0)
        convolution3d_smiv(umem, spad0, partial_layer, spad1);
    else
        convolution3d_smiv(umem, spad1, partial_layer, spad0);

    if (partial_layer.output_req == IO_DMA) {
        size_t num_output_pixels =
                partial_layer.outputs.rows * partial_layer.outputs.height *
                (partial_layer.outputs.cols + partial_layer.outputs.align_pad);
        if (input_in_spad0)
            dmaStore(host_results, spad1, num_output_pixels * sizeof(float));
        else
            dmaStore(host_results, spad0, num_output_pixels * sizeof(float));
    }

}

// Find a good way to pack the convolution into the accelerator.
static conv_cfg_t convolution_divide_work(layer_t* layers, int lnum) {
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
        PRINT_MSG_V("Entire input problem fits into the local memory.\n");
        init_work_cfg(&conv_cfgs, 1);
        conv_cfgs.iteration[0].rows = layers[lnum].inputs.rows;
        conv_cfgs.iteration[0].cols = layers[lnum].inputs.cols;
        conv_cfgs.iteration[0].height = layers[lnum].inputs.height;
        conv_cfgs.iteration[0].align_pad =
                calc_padding(conv_cfgs.iteration[0].cols, DATA_ALIGNMENT);
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
        PRINT_MSG_V("We can fit at least 2 unreduced input channels at once.\n");
        init_work_cfg(&conv_cfgs,
                      ceil((float)input_channels / max_channels_per_iter));
        unsigned total_channels = input_channels;
        for (unsigned i = 0; i < conv_cfgs.num_iterations; i++) {
            conv_cfgs.iteration[i].rows = layers[lnum].inputs.rows;
            conv_cfgs.iteration[i].cols = layers[lnum].inputs.cols;
            conv_cfgs.iteration[i].height =
                    min2(total_channels, max_channels_per_iter);
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

void standard_convolution_layer_impl(float* host_activations,
                                     float* host_weights,
                                     layer_t* layers,
                                     int lnum,
                                     float* host_result,
                                     device_t* device,
                                     sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int num_kerns = curr_layer.outputs.height;
    const int result_2d_size = result_height * (result_width + result_pad);
    ARRAY_4D(float, _result, host_result, num_kerns, result_height,
             result_width + result_pad);

#if DEBUG_LEVEL >= 1
    const int input_height = curr_layer.inputs.height;
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);
#endif

    conv_cfg_t conv_cfgs = convolution_divide_work(layers, lnum);
    printf("Standard convolution layer %d work configuration:\n", lnum);
    print_work_cfg(&conv_cfgs);

    // temp_result stores the partially reduced results of each iteration.
    size_t temp_result_size =
            result_2d_size * conv_cfgs.num_iterations * sizeof(float);
    float* temp_result = (float*)malloc_aligned(temp_result_size);

    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    bool use_acp_offload = (device->cpu_activation_func_offload == IO_ACP);

    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_activations", host_activations,
                       get_dims_size(&curr_layer.inputs));
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        for (int kern = 0; kern < num_kerns; kern++) {
            PRINT_MSG("Kernel %d\n", kern);
            PRINT_DEBUG4D(&_kernels[kern][0][0][0],
                          k_width,
                          k_width + k_pad,
                          input_height);
            unsigned start_chan = 0;
            float* result_loc = temp_result;
            for (unsigned iter = 0; iter < conv_cfgs.num_iterations; iter++) {
                PRINT_MSG("Iteration %d\n", iter);
                dims_t iter_cfg = conv_cfgs.iteration[iter];

                // Create a new layer description for this iteration.
                layer_t partial_layer = curr_layer;
                partial_layer.inputs.height = iter_cfg.height;
                partial_layer.outputs.height = iter_cfg.height;
                partial_layer.weights.height = iter_cfg.height;
                partial_layer.activation =
                        conv_cfgs.num_iterations > 1 || !do_hw_activation
                                ? NO_ACTIVATION
                                : curr_layer.activation;
                partial_layer.output_req = IO_NONE;
                INVOKE_KERNEL_PROF(kConvolutionHw, convolution_layer_hw,
                                   host_activations, host_weights, NULL, g_umem,
                                   g_spad0, g_spad1, true, layers,
                                   partial_layer, lnum, img, kern, start_chan);

                // Reduce the results.
                //
                // If the activation function is supported in hardware, then run
                // the standard reduction function with DMA. If the act func is
                // not supported, then use the ACP reduction impl, except if
                // the user specified to use DMA anyways.
                partial_layer.output_req = curr_layer.output_req;
                if (do_hw_activation || !use_acp_offload) {
                    MAP_ARRAY_TO_ACCEL(kReductionHw, "host_result", result_loc,
                                       temp_result_size);
                    INVOKE_KERNEL_PROF(kReductionHw, reduction_hw, g_spad0,
                                       g_spad1, g_umem, false, false,
                                       partial_layer, result_2d_size,
                                       result_loc);
                } else {
                    MAP_ARRAY_TO_ACCEL(kReductionHw, "acp_result", result_loc,
                                       temp_result_size);
                    INVOKE_KERNEL_PROF(kReductionHw, reduction_acp_hw, g_spad0,
                                       g_spad1, result_loc, false, false,
                                       partial_layer, result_2d_size);
                }

                result_loc += result_2d_size;
                start_chan += iter_cfg.height;
            }

            // Finish off the reduction here.
            if (conv_cfgs.num_iterations > 1) {
                result_loc = temp_result;

                int result_iter =
                        ceil(result_2d_size * conv_cfgs.num_iterations /
                             (float)SPAD_SIZE);
                assert(result_iter <= 1 &&
                       "Only support 1 last iteration of reduction!");
                int num_result_chans = min2(
                        conv_cfgs.num_iterations, SPAD_SIZE / result_2d_size);

                // Create a new layer description for this iteration.
                layer_t partial_layer = curr_layer;
                partial_layer.inputs.height = num_result_chans;
                partial_layer.outputs.height = 1;
                for (int iter = 0; iter < result_iter; iter++) {
                    PRINT_MSG("Final reduction round %d\n", iter);
                    if (do_hw_activation || !use_acp_offload) {
                        MAP_ARRAY_TO_ACCEL(kReductionHw, "host_result",
                                           result_loc, temp_result_size);
                        INVOKE_KERNEL_PROF(kReductionHw, reduction_hw, g_spad0,
                                           g_spad1, g_umem, false, true,
                                           partial_layer, result_2d_size,
                                           result_loc);
                    } else {
                        MAP_ARRAY_TO_ACCEL(kReductionHw, "acp_result",
                                           result_loc, result_2d_size);
                        INVOKE_KERNEL_PROF(kReductionHw, reduction_acp_hw,
                                           g_spad0, g_spad1, result_loc, false,
                                           true, partial_layer, result_2d_size);
                    }
                    result_loc += result_2d_size;
                }
            }

            // If the HW doesn't support the activation function, don't run the
            // activation function yet - we'll run it all at once when we're
            // done with all the kernels.

            memcpy(&_result[img][kern][0][0], temp_result,
                   result_2d_size * sizeof(float));
        }
    }
    free_work_cfg(&conv_cfgs);
    free(temp_result);
}

void depthwise_convolution_layer_impl(float* host_activations,
                                      float* host_weights,
                                      layer_t* layers,
                                      int lnum,
                                      float* host_result,
                                      device_t* device) {

    layer_t curr_layer = layers[lnum];
    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    ARRAY_3D(float, _result, host_result, result_height,
             result_width + result_pad);

#if DEBUG_LEVEL >= 1
    const int input_height = curr_layer.inputs.height;
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    ARRAY_3D(float, _kernels, host_weights, k_width, k_width + k_pad);
#endif

    conv_cfg_t conv_cfgs = convolution_divide_work(layers, lnum);
    printf("Depthwise convolution layer %d work configuration:\n", lnum);
    print_work_cfg(&conv_cfgs);

    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_activations", host_activations,
                       get_dims_size(&curr_layer.inputs));
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        float* current_result = &_result[img][0][0];
        unsigned start_chan = 0;
        for (unsigned iter = 0; iter < conv_cfgs.num_iterations; iter++) {
            PRINT_MSG("Iteration %d\n", iter);
            dims_t iter_cfg = conv_cfgs.iteration[iter];
            size_t current_iter_result_size =
                    iter_cfg.rows * iter_cfg.height *
                    (iter_cfg.cols + iter_cfg.align_pad);
            PRINT_MSG_V("Depthwise kernel channels %d-%d:\n", start_chan,
                        start_chan + iter_cfg.height);
            PRINT_DEBUG4D_V(&_kernels[start_chan][0][0], k_width,
                            k_width + k_pad, iter_cfg.height);

            // Create a new layer description for this iteration.
            layer_t partial_layer = curr_layer;
            partial_layer.inputs.height = iter_cfg.height;
            partial_layer.outputs.height = iter_cfg.height;
            partial_layer.weights.height = iter_cfg.height;
            // Unlike the standard convolution, filters are applied
            // channelwise, so we don't need to wait to do activation functions
            // until after the reductions (which don't exist in depthwise
            // convolution).
            partial_layer.activation =
                    !do_hw_activation ? NO_ACTIVATION : curr_layer.activation;
            // Always send data back.
            partial_layer.output_req = IO_DMA;
            MAP_ARRAY_TO_ACCEL(kConvolutionHw,
                               "host_results",
                               current_result,
                               current_iter_result_size);
            // The standard "kern" dimension is always 0, since the kernel
            // dimension is now the channel dimension.
            const int kern = 0;
            INVOKE_KERNEL_PROF(kConvolutionHw, convolution_layer_hw,
                               host_activations, host_weights, current_result,
                               g_umem, g_spad0, g_spad1, true, layers,
                               partial_layer, lnum, img, kern, start_chan);

            current_result += current_iter_result_size;
            start_chan += iter_cfg.height;
        }
    }

    // TODO: SMIV doesn't support activation functions on the CNN block, only
    // on the reduction block...so change it in the model :)
    // This is a hack to fake the idea that RELU is supported for
    // depthwise convolutions. SMIV's CNN block doesn't do activation functions
    // (this is handled by the reduction block) but depthwise convolutions
    // don't require reductions.
    if (layers[lnum].activation == RELU) {
        activation_fun(host_result, NUM_TEST_CASES,
                       get_dims_size(&layers[lnum].outputs), RELU, NULL);
    }
    free_work_cfg(&conv_cfgs);
}

#endif
