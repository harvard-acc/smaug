#ifndef _CORE_REORDER_OP_H_
#define _CORE_REORDER_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/operators/reorder_op_impl.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements a Tensor reordering operation to convert between different
 * DataLayouts.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class ReorderOp : public Operator {
   public:
    ReorderOp(const std::string& name,
              DataLayout _targetLayout,
              Workspace* workspace)
            : Operator(name, OpType::Reorder, workspace),
              targetLayout(_targetLayout) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    ReorderOp(const std::string& name, Workspace* workspace)
            : ReorderOp(name, DataLayout::UnknownLayout, workspace) {}

    DataLayout getTargetDataLayout() const { return targetLayout; }
    void setTargetLayout(DataLayout layout) { targetLayout = layout; }

    void run() override {
        auto stats = gem5::ScopedStats(
                stats::kReorderingStart, stats::kReorderingEnd);
        Tensor* input = getInput(Inputs);
        Tensor* output = getOutput(Outputs);
        DataLayout srcLayout = input->getShape().getLayout();
        if (srcLayout == DataLayout::NCHW) {
            if (targetLayout == DataLayout::NHWC) {
                convertNchwToNhwc(input, output);
            } else if (output->getShape().ndims() == 2) {
                flatten(input, output);
            }
        } else if (srcLayout == DataLayout::NHWC) {
            if (targetLayout == DataLayout::NCHW) {
                convertNhwcToNchw(input, output);
            } else if (output->getShape().ndims() == 2) {
                flatten(input, output);
            }
        } else if (input->getShape().ndims() == 3) {
            // NTC->NCT or NCT->NTC.
            assert(srcLayout == DataLayout::NCT &&
                           targetLayout == DataLayout::NTC ||
                   srcLayout == DataLayout::NTC &&
                           targetLayout == DataLayout::NCT &&
                           "Only NCT->NTC or NCT->NTC is supported for 3D "
                           "reorderings!");
            transpose3D(input, output);
        } else if (input->getShape().ndims() == 2) {
            if (srcLayout == targetLayout) {
                return;
            } else if (output->getShape().ndims() == 2) {
                transpose2D(input, output);
            } else {
                std::cerr << "Data layout reordering from "
                          << DataLayout_Name(srcLayout) << " to "
                          << DataLayout_Name(targetLayout)
                          << " is not supported!\n";
                exit(1);
            }
        }
    }

    bool validate() override {
        if (!Operator::validate())
            return false;
        DataLayout sourceLayout = inputs[Inputs]->getShape().getLayout();
        if (sourceLayout == DataLayout::UnknownLayout) {
            std::cerr << "[ERROR]: Reorder operation has unknown source "
                         "layout!\n";
            return false;
        }
        if (targetLayout == DataLayout::UnknownLayout) {
            std::cerr << "[ERROR]: Reorder operation has unknown target "
                         "layout!\n";
            return false;
        }
        if (sourceLayout == targetLayout) {
            std::cerr << "[ERROR]: Reorder operation does not change the data "
                         "layout!\n";
            return false;
        }
        return true;
    }

    TensorShape inferOutputShape() const {
        TensorShape inputShape = getInput(Inputs)->getShape();
        if (inputShape.ndims() == 4 && (targetLayout == DataLayout::NC ||
                                        targetLayout == DataLayout::CN)) {
            // Flatten a 4D tensor to 2D.
            std::vector<int> dims(2, 1);
            dims[0] = inputShape[0];
            for (int i = 1; i < inputShape.ndims(); ++i) {
                dims[1] *= inputShape[i];
            }
            return TensorShape(dims, targetLayout, Backend::Alignment);
        } else if (targetLayout == DataLayout::NC ||
                   targetLayout == DataLayout::CN) {
            // Transpose a 2D tensor.
            return TensorShape({ inputShape[1], inputShape[0] }, targetLayout,
                               Backend::Alignment);
        } else if (targetLayout == DataLayout::NCT ||
                   targetLayout == DataLayout::NTC) {
            // Transpose a 3D tensor.
            return TensorShape({ inputShape[0], inputShape[2], inputShape[1] },
                               targetLayout, Backend::Alignment);
        } else if (targetLayout == DataLayout::NCHW) {
            return TensorShape({ inputShape[0], inputShape[3], inputShape[1],
                                 inputShape[2] },
                               targetLayout, Backend::Alignment);
        } else if (targetLayout == DataLayout::NHWC) {
            return TensorShape({ inputShape[0], inputShape[2], inputShape[3],
                                 inputShape[1] },
                               targetLayout, Backend::Alignment);
        }
        return TensorShape();
    }

    void createOutputTensors() {
        assert(targetLayout != DataLayout::UnknownLayout &&
               "Cannot create output tensor with unknown target data layout!");
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(Outputs) = output;
    }
    void createAllTensors() override {
        createOutputTensors();
    }

   protected:
    enum { Inputs, kNumInputs };
    enum { Outputs, kNumOutputs };
    DataLayout targetLayout;
};

/** \ingroup FlattenOp
 *
 * Implements a flattening operation that squashes any Tensor DataLayout into
 * NC; that is, each batch in a Tensor is flattened into a single dimension.
 */
template <typename Backend>
class FlattenOp : public ReorderOp<Backend> {
   public:
    typedef ReorderOp<Backend> Parent;

    FlattenOp(const std::string& name, Workspace* workspace)
            : ReorderOp<Backend>(name, DataLayout::NC, workspace) {}
};

}  // namespace smaug

#endif
