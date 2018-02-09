#include <assert.h>
#include <string.h>

#include "arch/smiv_common.h"
#include "arch/smiv/dispatch_utils.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "core/smiv/smiv.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _convolution_options {
    int img;
    int kern;
    int start_chan;
    bool use_pipelined_dma;
} convolution_options;

typedef struct _reduction_options {
    bool input_in_spad0;
    bool use_pipelined_dma;
} reduction_options;

void reduction_hw_impl(float* host_inputs,
                       float* host_results,
                       float* local_inputs,
                       float* local_results,
                       layer_t partial_layer,
                       reduction_options* options) {
    size_t result_size =
            partial_layer.outputs.rows *
            (partial_layer.outputs.cols + partial_layer.outputs.align_pad);
    // This is only required if we want to initiate a final round of reduction.
    if (partial_layer.input_req == IO_DMA) {
        ASSERT(host_inputs && "host_inputs cannot be NULL for DMA!");
        size_t input_bytes =
                result_size * partial_layer.inputs.height * sizeof(float);
        setReadyBits(local_inputs, input_bytes, 0);
        dma_load_wrapper(local_inputs, host_inputs, input_bytes,
                         options->use_pipelined_dma);
    }

    reduction_smiv(local_inputs, partial_layer, local_results);

    if (partial_layer.output_req == IO_DMA) {
        ASSERT(host_results && "host_results cannot be NULL for DMA!");
        int result_bytes = result_size * sizeof(float);
        dma_store_wrapper(host_results, local_results, result_bytes,
                          options->use_pipelined_dma);
    }
}

void reduction_hw(float* dma_activations,
                  float* dma_results,
                  float* acp_activations,
                  float* acp_results,
                  float* cache_activations,
                  float* cache_results,
                  float* umem,
                  float* spad0,
                  float* spad1,
                  layer_t curr_layer,
                  access_config* access_config,
                  reduction_options* options) {

//=--------- Convenience macros for invoking the HW impl ---------------=//
//
// Each of these macros will call reduction_hw_impl() with a different name for
// the array arguments, based on the desired access mechanism for each. Since
// we name our variables in a consistent way - "mechanism_arrayname" - the
// macro can automatically form the correct variable name by macro
// concatentation.
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

#define REDUCTION_NO_DMA_IMPL(INPUT, LR)                                       \
    do {                                                                       \
        PRINT_MSG(#INPUT "-" #LR "\n");                                        \
        reduction_hw_impl(NULL, NULL, INPUT##_activations, LR##_results,       \
                          curr_layer, options);                                \
    } while (0)

#define REDUCTION_WITH_DMA_IMPL(HA, HR, LA, LR)                                \
    do {                                                                       \
        PRINT_MSG(#HA "-" #HR "-" #LA "-" #LR "\n");                           \
        reduction_hw_impl(                                                     \
                HA##_activations, HR##_results, LA, LR, curr_layer, options);  \
    } while (0)

// DMA used, with the input coming from either SPAD0 or SPAD1, but the output
// is going to a non-scratchpad location. Select the right input array with
// SELECT_SPAD0.
#define REDUCTION_WITH_DMA_SPAD_INPUT_IMPL(                                    \
        HA, HR, SPAD0, SPAD1, SELECT_SPAD0, LR)                                \
    do {                                                                       \
        if (SELECT_SPAD0) {                                                    \
            REDUCTION_WITH_DMA_IMPL(HA, HR, SPAD0, LR);                        \
        } else {                                                               \
            REDUCTION_WITH_DMA_IMPL(HA, HR, SPAD1, LR);                        \
        }                                                                      \
    } while (0)

// DMA used, with the output going to either SPAD0 or SPAD1, but the input is
// coming from a non-scratchpad location. Select the right one with
// SELECT_SPAD0.
#define REDUCTION_WITH_DMA_SPAD_OUTPUT_IMPL(                                   \
        HA, HR, LA, SPAD0, SPAD1, SELECT_SPAD0)                                \
    do {                                                                       \
        if (SELECT_SPAD0) {                                                    \
            REDUCTION_WITH_DMA_IMPL(HA, HR, LA, SPAD0);                        \
        } else {                                                               \
            REDUCTION_WITH_DMA_IMPL(HA, HR, LA, SPAD1);                        \
        }                                                                      \
    } while (0)

// DMA used, with both inputs and outputs going to/from scratchpads.
#define REDUCTION_WITH_DMA_SPAD_IO_IMPL(HA, HR, SPAD0, SPAD1, SELECT_SPAD0)    \
    do {                                                                       \
        if (SELECT_SPAD0) {                                                    \
            REDUCTION_WITH_DMA_IMPL(HA, HR, SPAD0, SPAD1);                     \
        } else {                                                               \
            REDUCTION_WITH_DMA_IMPL(HA, HR, SPAD1, SPAD0);                     \
        }                                                                      \
    } while (0)

    bool input_in_spad0 = options->input_in_spad0;
    if (DISPATCH_2(access_config, _DmaOrLocal, _DmaOrLocal)) {
        REDUCTION_WITH_DMA_SPAD_IO_IMPL(dma, dma, spad0, spad1, input_in_spad0);
    } else if (DISPATCH_2(access_config, _ACP, _ACP)) {
        REDUCTION_NO_DMA_IMPL(acp, acp);
    } else if (DISPATCH_2(access_config, _Cache, _Cache)) {
        REDUCTION_NO_DMA_IMPL(cache, cache);
    }
    // These selections only use _ACP or _Cache for the results.
    else if (DISPATCH_2(access_config, _DmaOrLocal, _ACP)) {
        REDUCTION_WITH_DMA_SPAD_INPUT_IMPL(
                dma, acp, spad0, spad1, input_in_spad0, acp_results);
    } else if (DISPATCH_2(access_config, _DmaOrLocal, _Cache)) {
        REDUCTION_WITH_DMA_SPAD_INPUT_IMPL(
                dma, cache, spad0, spad1, input_in_spad0, cache_results);
    }
    // These selections use DMA/None for the outputs.
    else if (DISPATCH_2(access_config, _ACP, _DmaOrLocal)) {
        REDUCTION_WITH_DMA_SPAD_OUTPUT_IMPL(
                acp, dma, acp_activations, spad0, spad1, input_in_spad0);
    } else if (DISPATCH_2(access_config, _Cache, _DmaOrLocal)) {
        REDUCTION_WITH_DMA_SPAD_OUTPUT_IMPL(
                cache, dma, cache_activations, spad0, spad1, input_in_spad0);
    }
    // Otherwise, give up.
    else {
        assert(false &&
               "This is an unsupported combination of access mechanisms!");
    }

#undef REDUCTION_WITH_DMA_SPAD_INPUT_IMPL
#undef REDUCTION_WITH_DMA_SPAD_OUTPUT_IMPL
#undef REDUCTION_WITH_DMA_SPAD_IO_IMPL
#undef REDUCTION_WITH_DMA_IMPL
#undef REDUCTION_NO_DMA_IMPL
}

// Main implementation of convolutional HW.
//
// Arguments:
//   dma_activations: The host address of the input activations.
//      **NOTE**: This should be the base of the entire input image, not the
//      base of a tiled section of the input.
//   dma_weights: The host address of the input weights.
//   dma_results: The host address of the input results.
//   local_activations: Pointer to inputs that the accelerator reads directly.
//   local_weights: Pointer to weights that the accelerator reads directly.
//   local_results: Pointer to results that the accelerator writes directly.
//   curr_layer: Description of this layer's shape and parameters.
//      **NOTE**: The input dims_t field in this describes the COMPLETE and
//      UNTILED dimensions input, whereas the dimensions for the weights and
//      outputs describe the TILED dimensions.
//   options: Additional options for this execution of convolution.
static void convolution_layer_hw_impl(float* dma_activations,
                                      float* dma_weights,
                                      float* dma_results,
                                      float* local_activations,
                                      float* local_weights,
                                      float* local_results,
                                      // TODO: make this a pointer instead.
                                      layer_t curr_layer,
                                      convolution_options* options) {
    // This is the full input height, NOT the tiled height!
    int total_input_height = curr_layer.inputs.height;
    int input_rows = curr_layer.inputs.rows;
    int input_cols = curr_layer.inputs.cols;
    int input_pad = curr_layer.inputs.align_pad;
    int start_chan = options->start_chan;
    int img = options->img;

    ARRAY_4D(float, _dma_activations, dma_activations, total_input_height,
             input_rows, input_cols + input_pad);

    // We should only DMA part of the weights.
    size_t num_weights =
            curr_layer.weights.rows * curr_layer.weights.height *
            (curr_layer.weights.cols + curr_layer.weights.align_pad);
    if (curr_layer.weights_req == IO_DMA) {
        ASSERT(dma_weights && "dma_weights cannot be NULL for DMA!");
        setReadyBits(local_weights, num_weights * sizeof(float), 0);
        dma_load_wrapper(local_weights, dma_weights,
                         num_weights * sizeof(float),
                         options->use_pipelined_dma);
    }
    if (curr_layer.input_req == IO_DMA) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        ASSERT(dma_activations && "dma_activations cannot be NULL for DMA!");
        size_t num_input_pixels =
                input_rows * total_input_height * (input_cols + input_pad);
        setReadyBits(local_activations, num_input_pixels * sizeof(float), 0);
        dma_load_wrapper(local_activations, &_dma_activations[img][0][0][0],
                         num_input_pixels * sizeof(float),
                         options->use_pipelined_dma);
    }

    convolution3d_smiv(local_activations, local_weights, curr_layer, start_chan,
                       local_results);

    if (curr_layer.output_req == IO_DMA) {
        ASSERT(dma_results && "dma_results cannot be NULL for DMA!");
        size_t num_output_pixels =
                curr_layer.outputs.rows * curr_layer.outputs.height *
                (curr_layer.outputs.cols + curr_layer.outputs.align_pad);
        dma_store_wrapper(dma_results, local_results,
                          num_output_pixels * sizeof(float),
                          options->use_pipelined_dma);
    }
}

// Main entry point of convolution HW.
//
// Unlike the other blocks, the conv block is restricted to storing inputs in
// the UMEM and the weights and outputs in the two scratchpads, so there is
// need to select the right spad.
//
static void convolution_layer_hw(float* dma_activations,
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
                                 layer_t curr_layer,
                                 access_config*  access_config,
                                 convolution_options* options) {
//=--------- Convenience macros for invoking the HW impl ---------------=//
//
// Because the convolutional block cannot mix the use of scratchpads and the
// umem, we don't need the macros to help us select which spad to use, which
// reduces the number of macros greatly.

// No DMA or scratchpads are used at all.
#define CONV3D_NO_DMA_IMPL(INPUT, WGT, LR)                                     \
    do {                                                                       \
        PRINT_MSG(#INPUT "-" #WGT "-" #LR "\n");                               \
        convolution_layer_hw_impl(NULL, NULL, NULL, INPUT##_activations,       \
                                  WGT##_weights, LR##_results, curr_layer,     \
                                  options);                                    \
    } while (0)

// Inputs can come from anywhere (dma, cache, or acp), and outputs can go
// anywhere.
#define CONV3D_WITH_DMA_IMPL(HA, HW, HR, LA, LW, LR)                           \
    do {                                                                       \
        PRINT_MSG(#HA "-" #HW "-" #HR "-" #LA "-" #LW "-" #LR "\n");           \
        convolution_layer_hw_impl(HA##_activations, HW##_weights,              \
                                  HR##_results, LA, LW, LR, curr_layer,        \
                                  options);                                    \
    } while (0)

    // These selections use the same mechanism all across.
    if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(dma, dma, dma, umem, spad0, spad1);
    } else if (DISPATCH_3(access_config, _ACP, _ACP, _ACP)) {
        CONV3D_NO_DMA_IMPL(acp, acp, acp);
    } else if (DISPATCH_3(access_config, _Cache, _Cache, _Cache)) {
        CONV3D_NO_DMA_IMPL(cache, cache, cache);
    }
    // These selections only use _ACP or _Cache for the results.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _ACP)) {
        CONV3D_WITH_DMA_IMPL(dma, dma, acp, umem, spad0, acp_results);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _DmaOrLocal, _Cache)) {
        CONV3D_WITH_DMA_IMPL(dma, dma, cache, umem, spad0, cache_results);
    }
    // These selections use DMA/None for the inputs.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _ACP, _ACP)) {
        CONV3D_WITH_DMA_IMPL(dma, acp, acp, umem, acp_weights, acp_results);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _Cache, _Cache)) {
        CONV3D_WITH_DMA_IMPL(
                dma, cache, cache, umem, cache_weights, cache_results);
    }
    // These selections use DMA/None for the inputs/outputs.
    //
    // NOTE: This scenario is currently not possible to specify via the model
    // configuration file.
    else if (DISPATCH_3(access_config, _DmaOrLocal, _ACP, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(dma, acp, dma, umem, acp_weights, spad1);
    } else if (DISPATCH_3(access_config, _DmaOrLocal, _Cache, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(dma, cache, dma, umem, cache_weights, spad1);
    }
    // These selections use DMA/None for the outputs.
    else if (DISPATCH_3(access_config, _ACP, _ACP, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(
                acp, acp, dma, acp_activations, acp_weights, spad1);
    } else if (DISPATCH_3(access_config, _Cache, _Cache, _DmaOrLocal)) {
        CONV3D_WITH_DMA_IMPL(
                cache, cache, dma, cache_activations, cache_weights, spad1);
    }
    // These selections only use DMA for the weights.
    else if (DISPATCH_3(access_config, _ACP, _DmaOrLocal, _ACP)) {
        CONV3D_WITH_DMA_IMPL(
                acp, dma, acp, acp_activations, spad0, acp_results);
    } else if (DISPATCH_3(access_config, _Cache, _DmaOrLocal, _Cache)) {
        CONV3D_WITH_DMA_IMPL(
                cache, dma, cache, cache_activations, spad0, cache_results);
    }
    // Otherwise, give up.
    else {
        assert(false &&
               "This is an unsupported combination of access mechanisms!");
    }

#undef CONV3D_WITH_DMA_IMPL
#undef CONV3D_NO_DMA_IMPL
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

    int max_channels_per_iter = SPAD_SIZE / output_channel_size;
    if (max_channels_per_iter >= 2) {
        PRINT_MSG_V("We can fit at least 2 unreduced input channels at once.\n");
        init_work_cfg(&conv_cfgs,
                      ceil((float)input_channels / max_channels_per_iter));
        int total_channels = input_channels;
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
    const layer_t curr_layer = layers[lnum];
    int result_height = curr_layer.outputs.rows;
    int result_width = curr_layer.outputs.cols;
    int result_pad = curr_layer.outputs.align_pad;
    int num_kerns = curr_layer.outputs.height;
    int input_height = curr_layer.inputs.height;
    int k_width = curr_layer.weights.cols;
    int k_pad = curr_layer.weights.align_pad;
    ARRAY_4D(float, _result, host_result, num_kerns, result_height,
             result_width + result_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);

    const int activations_size = INPUT_BYTES(layers, lnum);
    const int result_2d_size = result_height * (result_width + result_pad);
    const int weights_size = WEIGHT_BYTES(layers, lnum);

    conv_cfg_t conv_cfgs = convolution_divide_work(layers, lnum);
    INFO_MSG("Standard convolution layer %d work configuration:\n", lnum);
    print_work_cfg(&conv_cfgs);

    // temp_result stores the partially reduced results of each iteration.
    size_t temp_result_size =
            result_2d_size * conv_cfgs.num_iterations * sizeof(float);
    float* temp_result = (float*)malloc_aligned(temp_result_size);

    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    bool use_pipelined_dma = device->use_pipelined_dma;

    MAP_ARRAY_TO_ACCEL(kConvolutionHw,
                       get_host_inputs_var_name(curr_layer.input_req),
                       host_activations,
                       activations_size);
    if (curr_layer.input_req == IO_DMA) {
        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        flush_cache_range(host_activations, activations_size / sizeof(float));
        flush_cache_range(host_weights, weights_size / sizeof(float));
        end_profiling();
    }

    // Sampling: if set, only run up to the specified number of output
    // channels.  Set all remaining outputs to zero.
    const int sample_num_kerns = sampling_param->standard_conv_num_filters;
    const int num_kerns_to_simulate =
            sample_num_kerns == 0 ? num_kerns
                                  : min2(num_kerns, sample_num_kerns);
    bool is_sampled = num_kerns_to_simulate < num_kerns;
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        begin_profiling(__func__, lnum);
        if (is_sampled)
            set_profiling_type_sampled(num_kerns_to_simulate, num_kerns);

        for (int kern = 0; kern < num_kerns_to_simulate; kern++) {
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

                // Extraneous but just to reinforce the fact that inputs.height
                // refers to the COMPLETE, UNTILED height.
                partial_layer.inputs.height = curr_layer.inputs.height;
                // These height dimensions represent the TILED dimensions.
                partial_layer.outputs.height = iter_cfg.height;
                partial_layer.weights.height = iter_cfg.height;
                float* current_weights_loc = &_kernels[kern][start_chan][0][0];
                size_t weights_block_size =
                        get_dims_size(&partial_layer.weights);

                // For the 2D convolution, we don't want to do activation
                // functions or send data back.
                partial_layer.activation = NO_ACTIVATION;
                partial_layer.output_req = IO_NONE;

                // We send all the inputs to the UMEM in one go, so then we
                // can reuse them for later output channels.
                if (partial_layer.input_req == IO_DMA && kern > 0)
                    partial_layer.input_req = IO_NONE;

                convolution_options conv_options;
                conv_options.img = img;
                conv_options.kern = kern;
                conv_options.start_chan = start_chan;
                conv_options.use_pipelined_dma = use_pipelined_dma;
                access_config access_cfg =
                        layer_to_access_config(&partial_layer);
                MAP_ARRAY_TO_ACCEL(
                        kConvolutionHw,
                        get_host_weights_var_name(partial_layer.weights_req),
                        current_weights_loc, weights_block_size);
                INVOKE_KERNEL_PROF(kConvolutionHw,
                                   lnum,
                                   convolution_layer_hw,
                                   // DMA
                                   host_activations,
                                   current_weights_loc,
                                   NULL,
                                   // ACP
                                   host_activations,
                                   current_weights_loc,
                                   NULL,
                                   // Cache
                                   host_activations,
                                   current_weights_loc,
                                   NULL,
                                   g_umem,
                                   g_spad0,
                                   g_spad1,
                                   partial_layer,
                                   &access_cfg,
                                   &conv_options);

                // Reduce the results.
                //
                // If the activation function is supported in hardware, then run
                // the standard reduction function with DMA. If the act func is
                // not supported, then use the ACP reduction impl, except if
                // the user specified to use DMA anyways.
                partial_layer.input_req = IO_NONE;
                partial_layer.output_req = curr_layer.output_req;
                // For standard convolution, only do the activation in the
                // reduction block.
                partial_layer.activation =
                        conv_cfgs.num_iterations > 1 || !do_hw_activation
                                ? NO_ACTIVATION
                                : curr_layer.activation;
                MAP_ARRAY_TO_ACCEL(
                        kReductionHw,
                        get_host_results_var_name(partial_layer.output_req),
                        result_loc,
                        result_2d_size * sizeof(float));

                partial_layer.inputs.height = iter_cfg.height;
                reduction_options red_options;
                red_options.input_in_spad0 = false;
                red_options.use_pipelined_dma = device->use_pipelined_dma;
                access_cfg = layer_to_access_config(&partial_layer);
                INVOKE_KERNEL_PROF(kReductionHw,
                                   lnum,
                                   reduction_hw,
                                   // DMA
                                   NULL,
                                   result_loc,
                                   // ACP
                                   NULL,
                                   result_loc,
                                   // Cache
                                   NULL,
                                   result_loc,
                                   g_umem,
                                   g_spad0,
                                   g_spad1,
                                   partial_layer,
                                   &access_cfg,
                                   &red_options);

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

                // Create a new layer description for this iteration.
                layer_t partial_layer = curr_layer;
                partial_layer.inputs.height = (int)conv_cfgs.num_iterations;
                partial_layer.outputs.height = 1;

                PRINT_MSG("Final reduction round\n");
                if (partial_layer.output_req != IO_NONE) {
                    MAP_ARRAY_TO_ACCEL(
                            kReductionHw,
                            get_host_results_var_name(partial_layer.output_req),
                            result_loc,
                            temp_result_size);
                }
                if (do_hw_activation || partial_layer.output_req == IO_DMA) {
                    // Flush cache lines for temporary results.
                    begin_ignored_profiling(lnum);
                    flush_cache_range(
                            temp_result, temp_result_size / sizeof(float));
                    end_profiling();
                }
                partial_layer.input_req = device->cpu_default_offload;
                reduction_options red_options;
                red_options.input_in_spad0 = true;
                red_options.use_pipelined_dma = device->use_pipelined_dma;
                access_config access_cfg =
                        layer_to_access_config(&partial_layer);
                INVOKE_KERNEL_PROF(kReductionHw,
                                   lnum,
                                   reduction_hw,
                                   // DMA
                                   result_loc,
                                   result_loc,
                                   // ACP
                                   result_loc,
                                   result_loc,
                                   // Cache
                                   result_loc,
                                   result_loc,
                                   g_umem,
                                   g_spad0,
                                   g_spad1,
                                   partial_layer,
                                   &access_cfg,
                                   &red_options);
            }

            // If the HW doesn't support the activation function, don't run the
            // activation function yet - we'll run it all at once when we're
            // done with all the kernels.

            memcpy(&_result[img][kern][0][0], temp_result,
                   result_2d_size * sizeof(float));
        }
        end_profiling();

        // If we sampled the execution, set the remainder of the output to zero
        // to ensure we get the same results. Don't do this when simulating
        // though - even if we ran this part, it wouldn't be included as part
        // of the sampled time (we already called end_profiling() before this),
        // and it's more likely to throw off measurements. It doesn't represent any
        // part of the workload accurately either.
#ifndef GEM5_HARNESS
        if (is_sampled) {
            begin_ignored_profiling(lnum);
            memset(&_result[img][num_kerns_to_simulate][0][0], 0,
                   result_2d_size * sizeof(float) *
                           (num_kerns - num_kerns_to_simulate));
            end_profiling();
        }
#endif
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
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    const int activations_size = get_input_activations_size(&curr_layer);
    const int weights_size = get_num_weights_layer(layers, lnum);
    ARRAY_3D(float, _result, host_result, result_height,
             result_width + result_pad);
    ARRAY_3D(float, _kernels, host_weights, k_width, k_width + k_pad);

    conv_cfg_t conv_cfgs = convolution_divide_work(layers, lnum);
    INFO_MSG("Depthwise convolution layer %d work configuration:\n", lnum);
    print_work_cfg(&conv_cfgs);

    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);

    bool use_pipelined_dma = device->use_pipelined_dma;
    MAP_ARRAY_TO_ACCEL(kConvolutionHw,
                       get_host_inputs_var_name(curr_layer.input_req),
                       host_activations,
                       activations_size * sizeof(float));
    if (curr_layer.input_req == IO_DMA) {
        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        flush_cache_range(host_activations, activations_size);
        flush_cache_range(host_weights, weights_size);
        end_profiling();
    }
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
            // These are the UNTILED dimensions.
            partial_layer.inputs.height = iter_cfg.height;
            // These are the TILED dimensions.
            partial_layer.outputs.height = iter_cfg.height;
            partial_layer.weights.height = iter_cfg.height;
            float* current_weights_loc = &_kernels[start_chan][0][0];
            size_t weights_block_size = get_dims_size(&partial_layer.weights);

            // Unlike the standard convolution, filters are applied
            // channelwise, so we don't need to wait to do activation functions
            // until after the reductions (which don't exist in depthwise
            // convolution).
            partial_layer.activation =
                    !do_hw_activation ? NO_ACTIVATION : curr_layer.activation;
            // Always send data back.
            partial_layer.output_req = curr_layer.output_req;
            if (partial_layer.output_req != IO_NONE) {
                MAP_ARRAY_TO_ACCEL(
                        kConvolutionHw,
                        get_host_results_var_name(partial_layer.output_req),
                        current_result,
                        current_iter_result_size * sizeof(float));
            }
            convolution_options conv_options;
            conv_options.img = img;
            // The standard "kern" dimension is always 0, since the kernel
            // dimension is now the channel dimension.
            conv_options.kern = 0;
            conv_options.start_chan = start_chan;
            conv_options.use_pipelined_dma = use_pipelined_dma;
            access_config access_cfg = layer_to_access_config(&partial_layer);
            MAP_ARRAY_TO_ACCEL(
                    kConvolutionHw,
                    get_host_weights_var_name(partial_layer.weights_req),
                    current_weights_loc, weights_block_size);
            INVOKE_KERNEL_PROF(kConvolutionHw,
                               lnum,
                               convolution_layer_hw,
                               // DMA
                               host_activations,
                               current_weights_loc,
                               current_result,
                               // ACP
                               host_activations,
                               current_weights_loc,
                               current_result,
                               // Cache
                               host_activations,
                               current_weights_loc,
                               current_result,
                               g_umem,
                               g_spad0,
                               g_spad1,
                               partial_layer,
                               &access_cfg,
                               &conv_options);
            current_result += current_iter_result_size;
            start_chan += iter_cfg.height;
        }
    }
}
