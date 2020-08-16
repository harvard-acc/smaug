#ifndef _OPERATORS_POOLING_OP_H_
#define _OPERATORS_POOLING_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements a pooling operator.
 *
 * The pooling operator reduces the size of the input Tensor by applying a
 * windowed filter that reduces all elements in its field of view to a single
 * value.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class PoolingOp : public Operator {
   public:
    PoolingOp(const std::string& name, OpType _opType, Workspace* workspace)
            : Operator(name, _opType, workspace), poolingRowSize(0),
              poolingColSize(0), poolingRowStride(0), poolingColStride(0),
              sampling({ NoSampling, 1 }) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    std::pair<int, int> getPoolingSize() const {
        return std::make_pair(poolingRowSize, poolingColSize);
    }
    std::pair<int, int> getPoolingStride() const {
        return std::make_pair(poolingRowStride, poolingColStride);
    }

    void setPoolingSize(int rowSize, int colSize) {
        poolingRowSize = rowSize;
        poolingColSize = colSize;
    }

    void setPoolingStride(int rowStride, int colStride) {
        poolingRowStride = rowStride;
        poolingColStride = colStride;
    }

    bool validate() override {
        return (poolingColSize > 0 && poolingRowStride > 0 &&
                poolingColStride > 0 && Operator::validate());
    }

    int getNumOfmaps() const {
        Tensor* input = getInput(0);
        assert(input && "Unable to find input for pooling layer!");
        const TensorShape& inputShape = inputs.at(Inputs)->getShape();
        bool isNCHW = inputShape.getLayout() == DataLayout::NCHW;
        int chanIdx = isNCHW ? 1 : 3;
        return input->dim(chanIdx);
    }

    TensorShape inferOutputShape() const {
        const TensorShape& inputShape = inputs.at(Inputs)->getShape();
        bool isNCHW = inputShape.getLayout() == DataLayout::NCHW;
        int inputRows = isNCHW ? inputShape[2] : inputShape[1];
        int inputCols = isNCHW ? inputShape[3] : inputShape[2];
        int inputChans = isNCHW ? inputShape[1] : inputShape[3];
        int outputRows = calcOutputRows(inputRows);
        int outputCols = calcOutputCols(inputCols);
        assert(outputRows > 0 && outputCols > 0 &&
               "Pooling layer field size exceeds the input image dimensions!");
        if (isNCHW) {
            return TensorShape(
                    { inputShape[0], inputChans, outputRows, outputCols },
                    inputShape.getLayout(), Backend::Alignment);
        } else {
            return TensorShape(
                    { inputShape[0], outputRows, outputCols, inputChans },
                    inputShape.getLayout(), Backend::Alignment);
        }
    }

    void createOutputTensors() {
        if (outputs.at(Outputs))
            return;
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(Outputs) = output;
    }

    void createAllTensors() override { createOutputTensors(); }

    bool isSamplingSupported() const override { return true; }
    void setSamplingInfo(const SamplingInfo& _sampling) override {
        sampling = _sampling;
    }

   protected:
    int calcOutputRows(int inputRows) const {
        return computeOutputDim(inputRows, poolingRowSize, poolingRowStride);
    }
    int calcOutputCols(int inputCols) const {
        return computeOutputDim(inputCols, poolingColSize, poolingColStride);
    }

    int computeOutputDim(int inputDims, int poolSize, int poolStride) const {
        return (inputDims - poolSize) / poolStride + 1;
    }

    enum { Inputs, kNumInputs };
    enum { Outputs, kNumOutputs };

    int poolingRowSize;
    int poolingColSize;
    int poolingRowStride;
    int poolingColStride;
    SamplingInfo sampling;
};

/** \ingroup Operators
 * Implements the max-pooling operator.
 */
template <typename Backend>
class MaxPoolingOp : public PoolingOp<Backend> {
   protected:
    typedef PoolingOp<Backend> Parent;

   public:
    MaxPoolingOp(const std::string& name, Workspace* workspace)
            : PoolingOp<Backend>(name, OpType::MaxPooling, workspace) {}
    void run() override{};
};

/** \ingroup Operators
 * Implements the arithmetic-average-pooling operator.
 */
template <typename Backend>
class AvgPoolingOp : public PoolingOp<Backend> {
   protected:
    typedef PoolingOp<Backend> Parent;

   public:
    AvgPoolingOp(const std::string& name, Workspace* workspace)
            : PoolingOp<Backend>(name, OpType::AveragePooling, workspace) {}
    void run() override{};
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(MaxPoolingOp, ReferenceBackend);
REGISTER_SPECIAL_OP(AvgPoolingOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
