#include <string.h>

#include "arch/smiv/dispatch_utils.h"
#include "arch/smiv_common.h"
#include "core/smv/params.h"
#include "core/smv/smv.h"
#include "utility/data_layout_conversion.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

static void convolution_layer_smv_hw(float* host_activations,
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
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = partial_layer.inputs.align_pad;
    const int k_width = partial_layer.weights.cols;
    const int k_height = partial_layer.weights.height;
    const int k_pad = partial_layer.weights.align_pad;
    ARRAY_4D(float, _a, host_activations, input_height + input_pad, input_rows,
             input_cols);
    ARRAY_4D(float, _kernels, host_weights, k_width, k_width, k_height + k_pad);
    // We should only DMA part of the weights.
    size_t num_weights =
            NUM_PE_INSTS * partial_layer.weights.rows *
            (partial_layer.weights.height + partial_layer.weights.align_pad) *
            partial_layer.weights.cols;
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
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                partial_layer.inputs.rows *
                (partial_layer.inputs.height + partial_layer.inputs.align_pad) *
                (partial_layer.inputs.cols);
        setReadyBits(umem, num_input_pixels * sizeof(float), 0);
        dmaLoad(umem, &_a[img][0][0][0], num_input_pixels * sizeof(float));
    }

    if (input_in_spad0)
        convolution3d_smv(umem, spad0, partial_layer, start_chan, spad1);
    else
        convolution3d_smv(umem, spad1, partial_layer, start_chan, spad0);

    if (partial_layer.output_req == IO_DMA) {
        size_t num_output_pixels =
                partial_layer.outputs.rows * NUM_PE_INSTS *
                (partial_layer.outputs.cols + partial_layer.outputs.align_pad);
        if (input_in_spad0)
            dmaStore(host_results, spad1, num_output_pixels * sizeof(float));
        else
            dmaStore(host_results, spad0, num_output_pixels * sizeof(float));
    }
}

static conv_cfg_t convolution_divide_work(layer_t* layers, int lnum) {
    conv_cfg_t conv_cfgs;
    unsigned total_input_bytes = INPUT_BYTES(layers, lnum) / NUM_TEST_CASES;

    if (total_input_bytes > UMEM_SIZE) {
        printf("A single input image exceeds the capacity of the UMEM, which "
               "is not supported!\n");
        assert(false);
    }
    //	#ifdef NHWC_COMPUTE
    // Divide the problem spatially to do multiple calls to the HW ACCEL.
    // For now, we shall assume we are going to compute one output filter per
    // H.W ioctl call. And then loop across the different filters.
    unsigned total_output_bytes =
            get_dims_size(&layers[lnum].outputs) * sizeof(float);
    if (total_output_bytes <= SPAD_SIZE) {
        PRINT_MSG_V("Entire input problem fits into the local memory.\n");
        init_work_cfg(&conv_cfgs, 1);
        conv_cfgs.iteration[0].rows = layers[lnum].inputs.rows;
        conv_cfgs.iteration[0].cols = layers[lnum].inputs.cols;
        conv_cfgs.iteration[0].height = layers[lnum].inputs.height;
        conv_cfgs.iteration[0].align_pad =
                calc_padding(conv_cfgs.iteration[0].cols, DATA_ALIGNMENT);
        return conv_cfgs;
    } else {
        printf("No tiling is supported!\n");
        assert(false);
        return conv_cfgs;
    }
}

void standard_convolution_layer_smv_impl(float* host_activations,
                                         float* host_weights,
                                         layer_t* layers,
                                         int lnum,
                                         float* host_result,
                                         device_t* device,
                                         sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    const int result_height = curr_layer.outputs.height;
    const int result_length = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int num_kerns = curr_layer.outputs.height;
    const int input_height = curr_layer.inputs.height;
    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    const int result_3d_size =
            result_length * (result_width + result_pad) * result_height;
    const int output_chan_batch_size  = NUM_PE_INSTS;
    const int single_kernel_size =
            get_num_weights_layer(layers, lnum) * sizeof(float);
    float* nhwc_activations = NULL;
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            host_activations, NUM_TEST_CASES, curr_layer.inputs, DATA_ALIGNMENT,
            &nhwc_activations);
    ARRAY_4D(float, _result, host_result, result_height, result_length,
             result_width + result_pad);
    ARRAY_4D(float, _kernels, host_weights, input_height, k_width,
             k_width + k_pad);

    conv_cfg_t conv_cfgs = convolution_divide_work(layers, lnum);
    print_work_cfg(&conv_cfgs);

    // conv_output stores results of convolution for eack kernel.
    // For now keeping num_iterations to 1 but will actually increase once I
    // finish the tiling code properly.
    size_t conv_output_size =
            result_3d_size * conv_cfgs.num_iterations * sizeof(float);
    float* conv_output = (float*)malloc_aligned(conv_output_size);
    bool do_hw_activation = device->use_hw_activation_func &&
                            is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);

    MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_activations", nhwc_activations,
                       get_dims_size(&curr_layer.inputs) * sizeof(float));

    printf("NWHC activations\n");
    print_debug4d(nhwc_activations, activations_nhwc.rows,
                  activations_nhwc.cols, activations_nhwc.height);
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
        for (int kern = 0;
             kern < num_kerns_to_simulate / output_chan_batch_size;
             kern++) {
            // Convert Weights to NHWC.
            float* nhwc_weights = NULL;
            dims_t weights_nhwc = convert_nchw_to_nhwc(
                    &_kernels[kern * output_chan_batch_size][0][0][0],
                    output_chan_batch_size, curr_layer.weights, DATA_ALIGNMENT,
                    &nhwc_weights);
            printf("NWHC weights\n");
            print_debug4d(nhwc_weights, weights_nhwc.rows, weights_nhwc.cols,
                          weights_nhwc.height);
            MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_weights", nhwc_weights,
                               output_chan_batch_size * single_kernel_size);
            unsigned start_chan = 0;
            for (unsigned iter = 0; iter < conv_cfgs.num_iterations; iter++) {
                dims_t iter_cfg = conv_cfgs.iteration[iter];
                layer_t partial_layer = curr_layer;
                partial_layer.outputs.height = output_chan_batch_size;
                partial_layer.weights.height = iter_cfg.height;
                partial_layer.inputs.align_pad =
                        calc_padding(curr_layer.inputs.height, DATA_ALIGNMENT);
                partial_layer.weights.align_pad =
                        calc_padding(curr_layer.weights.height, DATA_ALIGNMENT);
                partial_layer.outputs.align_pad =
                        calc_padding(curr_layer.outputs.cols, DATA_ALIGNMENT);
                partial_layer.activation =
                        conv_cfgs.num_iterations > 1 || !do_hw_activation
                                ? NO_ACTIVATION
                                : curr_layer.activation;
                partial_layer.output_req = IO_DMA;
                partial_layer.input_req = (kern == 0) ? IO_DMA : IO_NONE;
                MAP_ARRAY_TO_ACCEL(kConvolutionHw, "host_results", conv_output,
                                   conv_output_size);
                INVOKE_KERNEL_PROF(kConvolutionHw, lnum, convolution_layer_smv_hw,
                                   nhwc_activations, nhwc_weights, conv_output,
                                   g_umem, g_spad0, g_spad1, true, layers,
                                   partial_layer, lnum, img, 0, start_chan);
            }
            memcpy(&_result[img][kern][0][0], conv_output,
                   result_3d_size * sizeof(float));
        }
        end_profiling();
    }
    free(nhwc_activations);
    free_work_cfg(&conv_cfgs);
    free(conv_output);
}
