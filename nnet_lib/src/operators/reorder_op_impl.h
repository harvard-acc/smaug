#include "core/tensor.h"

namespace smaug {

template <typename DType, typename Backend>
void convertNchwToNhwcImpl(Tensor<Backend>* input, Tensor<Backend>* output) {
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

template <typename DType, typename Backend>
void convertNhwcToNchwImpl(Tensor<Backend>* input, Tensor<Backend>* output) {
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

template <typename DType, typename Backend>
void flattenImpl(Tensor<Backend>* input, Tensor<Backend>* output) {
    TensorIndexIterator inputIdx = input->startIndex();
    TensorIndexIterator outputIdx = output->startIndex();
    const TensorShape& inputShape = input->getShape();
    DType* inputData = input->template data<DType>();
    DType* outputData = output->template data<DType>();
    for (int n = 0; n < inputShape[0]; n++) {
        int out_i = 0;
        // At this point, it doesn't matter whether the layout is NCHW or NHWC.
        // We just need to flatten the HWC part, which is dictated by the size
        // of each dimension and not the logical meaning of each dim.
        for (int i = 0; i < inputShape[1]; i++) {
            for (int j = 0; j < inputShape[2]; j++) {
                for (int k = 0; k < inputShape[3]; k++) {
                    outputData[outputIdx(n, out_i++)] =
                            inputData[inputIdx(n, i, j, k)];
                }
            }
        }
    }
}

template <typename Backend>
void convertNchwToNhwc(Tensor<Backend>* input, Tensor<Backend>* output) {
    DataType datatype = input->getDataType();
    assert(input->ndims() == output->ndims() && input->ndims() == 4);
    switch (datatype) {
        case Float16:
            convertNchwToNhwcImpl<float16, Backend>(input, output);
            return;
        case Float32:
            convertNchwToNhwcImpl<float, Backend>(input, output);
            return;
        case Float64:
            convertNchwToNhwcImpl<double, Backend>(input, output);
            return;
        case Int32:
            convertNchwToNhwcImpl<int, Backend>(input, output);
            return;
        case Int64:
            convertNchwToNhwcImpl<int64_t, Backend>(input, output);
            return;
        default:
            assert(false && "Unknown data format!");
    }
}

template <typename Backend>
void convertNhwcToNchw(Tensor<Backend>* input, Tensor<Backend>* output) {
    DataType datatype = input->getDataType();
    assert(input->ndims() == output->ndims() && input->ndims() == 4);
    switch (datatype) {
        case Float16:
            convertNhwcToNchwImpl<float16, Backend>(input, output);
            return;
        case Float32:
            convertNhwcToNchwImpl<float, Backend>(input, output);
            return;
        case Float64:
            convertNhwcToNchwImpl<double, Backend>(input, output);
            return;
        case Int32:
            convertNhwcToNchwImpl<int, Backend>(input, output);
            return;
        case Int64:
            convertNhwcToNchwImpl<int64_t, Backend>(input, output);
            return;
        default:
            assert(false && "Unknown data format!");
    }
}

template <typename Backend>
void flatten(Tensor<Backend>* input, Tensor<Backend>* output) {
    DataType datatype = input->getDataType();
    assert(input->ndims() == 4 && output->ndims() == 2);
    switch (datatype) {
        case Float16:
            flattenImpl<float16, Backend>(input, output);
            return;
        case Float32:
            flattenImpl<float, Backend>(input, output);
            return;
        case Float64:
            flattenImpl<double, Backend>(input, output);
            return;
        case Int32:
            flattenImpl<int, Backend>(input, output);
            return;
        case Int64:
            flattenImpl<int64_t, Backend>(input, output);
            return;
        default:
            assert(false && "Unknown data format!");
    }
}

}  // namespace smaug
