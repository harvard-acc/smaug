#include "smaug/core/tensor.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

template <typename DType>
void convertNchwToNhwcImpl(Tensor* input, Tensor* output) {
    TensorIndexIterator inputIdx = input->startIndex();
    TensorIndexIterator outputIdx = output->startIndex();
    const TensorShape& inputShape = input->getShape();
    DType* inputData = input->template data<DType>();
    DType* outputData = output->template data<DType>();
    for (int n = 0; n < inputShape[0]; n++) {
        for (int c = 0; c < inputShape[1]; c++) {
            for (int h = 0; h < inputShape[2]; h++) {
                for (int w = 0; w < inputShape[3]; w++) {
                    outputData[outputIdx(n, h, w, c)] =
                            inputData[inputIdx(n, c, h, w)];
                }
            }
        }
    }
}

template <typename DType>
void convertNhwcToNchwImpl(Tensor* input, Tensor* output) {
    TensorIndexIterator inputIdx = input->startIndex();
    TensorIndexIterator outputIdx = output->startIndex();
    const TensorShape& inputShape = input->getShape();
    DType* inputData = input->template data<DType>();
    DType* outputData = output->template data<DType>();
    for (int n = 0; n < inputShape[0]; n++) {
        for (int h = 0; h < inputShape[1]; h++) {
            for (int w = 0; w < inputShape[2]; w++) {
                for (int c = 0; c < inputShape[3]; c++) {
                    outputData[outputIdx(n, c, h, w)] =
                            inputData[inputIdx(n, h, w, c)];
                }
            }
        }
    }
}

template <typename DType>
void flattenImpl(Tensor* input, Tensor* output) {
    TensorIndexIterator inputIdx = input->startIndex();
    TensorIndexIterator outputIdx = output->startIndex();
    const TensorShape& inputShape = input->getShape();
    const TensorShape& outputShape = output->getShape();
    DType* inputData = input->template data<DType>();
    DType* outputData = output->template data<DType>();
    bool targetNC = outputShape.getLayout() == NC;
    for (int n = 0; n < inputShape[0]; n++) {
        int out_i = 0;
        // At this point, it doesn't matter whether the layout is NCHW or NHWC.
        // We just need to flatten the HWC part, which is dictated by the size
        // of each dimension and not the logical meaning of each dim.
        for (int i = 0; i < inputShape[1]; i++) {
            for (int j = 0; j < inputShape[2]; j++) {
                for (int k = 0; k < inputShape[3]; k++) {
                    if (targetNC) {
                        outputData[outputIdx(n, out_i++)] =
                                inputData[inputIdx(n, i, j, k)];
                    } else {
                        outputData[outputIdx(out_i++, n)] =
                                inputData[inputIdx(n, i, j, k)];
                    }
                }
            }
        }
    }
}

template <typename DType>
void transpose3DImpl(Tensor* input, Tensor* output) {
    TensorIndexIterator inputIdx = input->startIndex();
    TensorIndexIterator outputIdx = output->startIndex();
    const TensorShape& inputShape = input->getShape();
    auto inputData = input->template data<DType>();
    auto outputData = output->template data<DType>();
    for (int i = 0; i < inputShape[0]; i++) {
        for (int j = 0; j < inputShape[1]; j++) {
            for (int k = 0; k < inputShape[2]; k++) {
                outputData[outputIdx(i, k, j)] = inputData[inputIdx(i, j, k)];
            }
        }
    }
}

template <typename DType>
void transpose2DImpl(Tensor* input, Tensor* output) {
    TensorIndexIterator inputIdx = input->startIndex();
    TensorIndexIterator outputIdx = output->startIndex();
    const TensorShape& inputShape = input->getShape();
    auto inputData = input->template data<DType>();
    auto outputData = output->template data<DType>();
    for (int n = 0; n < inputShape[0]; n++) {
        for (int c = 0; c < inputShape[1]; c++) {
            outputData[outputIdx(c, n)] = inputData[inputIdx(n, c)];
        }
    }
}

void convertNchwToNhwc(Tensor* input, Tensor* output);

void convertNhwcToNchw(Tensor* input, Tensor* output);

void flatten(Tensor* input, Tensor* output);

void transpose3D(Tensor* input, Tensor* output);

void transpose2D(Tensor* input, Tensor* output);

}  // namespace smaug
