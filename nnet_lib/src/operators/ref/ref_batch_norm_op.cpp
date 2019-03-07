#include "core/backend.h"
#include "operators/common.h"
#include "operators/batch_norm_op.h"
#include "utility/debug_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// 1/sqrt(var + eps) is precomputed to avoid having to run a sqrt and division
// in the ASIC.
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

// For batch norm following a FC layer, we have one pair of gamma/beta weights
// per activation.
void ref_batch_norm_f32_post_fc(float* inputs,
                                float* mean,
                                float* variance,
                                float* gamma,
                                float* beta,
                                float* result,
                                int input_nums,
                                int input_size,
                                int input_pad) {
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
}

// For batch norm following a convolutional/pooling layer, we only have a
// gamma/beta per output feature map, not per activation.
void ref_batch_norm_f32_nchw_post_conv(float* inputs,
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
                                       int wgt_pad) {
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

    if (isPostConv) {
        assert(inputShape.getLayout() == DataLayout::NCHW);
        assert(outputShape.getLayout() == DataLayout::NCHW);
        ref_batch_norm_f32_nchw_post_conv(
                input->data<float>(), mean->data<float>(),
                variance->data<float>(), gamma->data<float>(),
                beta->data<float>(), output->data<float>(), inputShape[0],
                inputShape[1], inputShape[2], inputShape[3], 0, 0);
    } else {
        assert(inputShape.getLayout() == DataLayout::NC);
        assert(outputShape.getLayout() == DataLayout::NC);
        ref_batch_norm_f32_post_fc(input->data<float>(), mean->data<float>(),
                                   variance->data<float>(),
                                   gamma->data<float>(), beta->data<float>(),
                                   output->data<float>(), inputShape[0],
                                   inputShape[1], 0);
    }
}

}  // namespace smaug
