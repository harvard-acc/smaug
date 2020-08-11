#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/batch_norm_op.h"
#include "smaug/operators/ref/ref_activation_fun_op.h"
#include "smaug/utility/debug_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * Batch normalizes one input value.
 *
 * @param input Input activation.
 * @param mean Batch mean
 * @param recip_sqrt_var 1/sqrt(var + eps), which is precomputed to avoid
 * having to run a sqrt and division in the ASIC.
 * @param gamma Gamma parameter.
 * @param beta Beta parameter.
 */
ALWAYS_INLINE
float batch_norm_op(float input,
                    float mean,
                    float recip_sqrt_var,
                    float gamma,
                    float beta) {
    float scale = recip_sqrt_var * gamma;
    float shift = input - mean;
    return shift * scale + beta;
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of batch normalization following a
 * fully-connected layer.
 *
 * In this case, we have one pair of gamma/beta weights per activation.
 */
void ref_batch_norm_post_fc(float* inputs,
                            float* mean,
                            float* variance,
                            float* gamma,
                            float* beta,
                            float* result,
                            int input_nums,
                            int input_size,
                            int input_pad,
                            activation_type act_function,
                            activation_param_t act_params) {
    int inputs_size = input_nums * (input_size + input_pad);
    int kernel_size = inputs_size;
    int result_size = inputs_size;
    dmaLoad(inputs, inputs, inputs_size * sizeof(float));
    dmaLoad(mean, mean, kernel_size * sizeof(float));
    dmaLoad(variance, variance, kernel_size * sizeof(float));
    dmaLoad(gamma, gamma, kernel_size * sizeof(float));
    dmaLoad(beta, beta, kernel_size * sizeof(float));

    ARRAY_2D(float, _inputs, inputs, input_size + input_pad);
    ARRAY_2D(float, _result, result, input_size + input_pad);

    bn_batch:
    for (int i = 0; i < input_nums; i++) {
        bn_input:
        for (int j = 0; j < input_size; j++) {
            _result[i][j] = batch_norm_op(
                    _inputs[i][j], mean[j], variance[j], gamma[j], beta[j]);
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of batch normalization following a
 * convolutional/pooling layer on NCHW data.
 *
 * After conv/pooling, we only have a gamma/beta per output feature map, not
 * per activation.
 */
void ref_batch_norm_nchw_post_conv(float* inputs,
                                   float* mean,
                                   float* variance,
                                   float* gamma,
                                   float* beta,
                                   float* result,
                                   int img_nums,
                                   int img_chans,
                                   int img_rows,
                                   int img_cols,
                                   int img_pad,
                                   int wgt_pad,
                                   activation_type act_function,
                                   activation_param_t act_params) {
    int input_size = img_nums * img_chans * img_rows * (img_cols + img_pad);
    int kernel_size = img_chans;
    int result_size = input_size;
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    dmaLoad(mean, mean, kernel_size * sizeof(float));
    dmaLoad(variance, variance, kernel_size * sizeof(float));
    dmaLoad(gamma, gamma, kernel_size * sizeof(float));
    dmaLoad(beta, beta, kernel_size * sizeof(float));

    ARRAY_4D(float, _inputs, inputs, img_chans, img_rows, img_cols + img_pad);
    ARRAY_4D(float, _result, result, img_chans, img_rows, img_cols + img_pad);

    bn_batch:
    for (int i = 0; i < img_nums; i++) {
        bn_chan:
        for (int h = 0; h < img_chans; h++) {
            float mean_val = mean[h];
            float recip_sqrt_var_val = variance[h];
            float gamma_val = gamma[h];
            float beta_val = beta[h];

            bn_row:
            for (int r = 0; r < img_rows; r++) {
                bn_col:
                for (int c = 0; c < img_cols; c++) {
                    _result[i][h][r][c] = batch_norm_op(_inputs[i][h][r][c],
                                                        mean_val,
                                                        recip_sqrt_var_val,
                                                        gamma_val,
                                                        beta_val);
                }
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of batch normalization following a
 * convolutional/pooling layer on NHWC data.
 *
 * After conv/pooling, we only have a gamma/beta per output feature map, not
 * per activation.
 */
void ref_batch_norm_nhwc_post_conv(float* inputs,
                                   float* mean,
                                   float* variance,
                                   float* gamma,
                                   float* beta,
                                   float* result,
                                   int img_nums,
                                   int img_rows,
                                   int img_cols,
                                   int img_chans,
                                   int img_pad,
                                   int wgt_pad,
                                   activation_type act_function,
                                   activation_param_t act_params) {
    int input_size = img_nums * img_rows * img_cols * (img_chans + img_pad);
    int kernel_size = img_chans;
    int result_size = input_size;
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    dmaLoad(mean, mean, kernel_size * sizeof(float));
    dmaLoad(variance, variance, kernel_size * sizeof(float));
    dmaLoad(gamma, gamma, kernel_size * sizeof(float));
    dmaLoad(beta, beta, kernel_size * sizeof(float));

    ARRAY_4D(float, _inputs, inputs, img_rows, img_cols, img_chans + img_pad);
    ARRAY_4D(float, _result, result, img_rows, img_cols, img_chans + img_pad);

    bn_batch:
    for (int i = 0; i < img_nums; i++) {
        bn_chan:
        for (int h = 0; h < img_chans; h++) {
            float mean_val = mean[h];
            float recip_sqrt_var_val = variance[h];
            float gamma_val = gamma[h];
            float beta_val = beta[h];
            bn_row:
            for (int r = 0; r < img_rows; r++) {
                bn_col:
                for (int c = 0; c < img_cols; c++) {
                    _result[i][r][c][h] = batch_norm_op(_inputs[i][r][c][h],
                                                        mean_val,
                                                        recip_sqrt_var_val,
                                                        gamma_val,
                                                        beta_val);
                }
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void BatchNormOp<ReferenceBackend>::run() {
    auto input = getInput(Inputs);
    auto mean = getInput(Mean);
    auto variance = getInput(Variance);
    auto gamma = getInput(Gamma);
    auto beta = getInput(Beta);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = mean->getShape();
    const TensorShape& outputShape = output->getShape();
    bool isPostConv = (input->ndims() == 4);
    dout(2) << *mean << "\n";
    dout(2) << *variance<< "\n";
    dout(2) << *gamma << "\n";
    dout(2) << *beta << "\n";

    float* inputData = input->data<float>();
    float* meanData = mean->data<float>();
    float* varianceData = variance->data<float>();
    float* gammaData = gamma->data<float>();
    float* betaData = beta->data<float>();
    float* outputData = output->data<float>();
    mapArrayToAccel(ref::kBatchNormHw, "inputs", inputData,
                    inputShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kBatchNormHw, "mean", meanData,
                    kernelShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kBatchNormHw, "variance", varianceData,
                    kernelShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kBatchNormHw, "gamma", gammaData,
                    kernelShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kBatchNormHw, "beta", betaData,
                    kernelShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kBatchNormHw, "result", outputData,
                    outputShape.storageSize() * sizeof(float));
    if (isPostConv) {
        bool isNCHW = input->getShape().getLayout() == NCHW;
        auto func = isNCHW ? ref_batch_norm_nchw_post_conv
                           : ref_batch_norm_nhwc_post_conv;
        invokeKernel(ref::kBatchNormHw, func, inputData, meanData, varianceData,
                     gammaData, betaData, outputData, inputShape[0],
                     inputShape[1], inputShape[2], inputShape[3],
                     inputShape.getPadding(3), kernelShape.getPadding(3),
                     actInfo.function, actInfo.params);
    } else {
        assert(inputShape.getLayout() == DataLayout::NC);
        assert(outputShape.getLayout() == DataLayout::NC);
        invokeKernel(ref::kBatchNormHw, ref_batch_norm_post_fc, inputData,
                     meanData, varianceData, gammaData, betaData, outputData,
                     inputShape[0], inputShape[1], inputShape.getPadding(1),
                     actInfo.function, actInfo.params);
    }
}

}  // namespace smaug
