#ifndef _CORE_BACKEND_H_
#define _CORE_BACKEND_H_

#include <string>

// These are compile-time switches that selectively build a copy of SMAUG with
// a particular backend.
#define REFERENCE 0
#define SMVBACKEND 1

namespace smaug {

enum BackendName {
    Reference = REFERENCE,
    Smv = SMVBACKEND,
    UnknownBackend,
};

class Workspace;
template <typename Backend> class ConvolutionOp;
template <typename Backend> class DataOp;
template <typename Backend> class DepthwiseConvolutionOp;
template <typename Backend> class MaxPoolingOp;
template <typename Backend> class AvgPoolingOp;
template <typename Backend> class InnerProductOp;
template <typename Backend> class SoftmaxOp;
template <typename Backend> class FlattenOp;
template <typename Backend> class BatchNormOp;
template <typename Backend> class EltwiseAddOp;
template <typename Backend> class ReluOp;
template <typename Backend> class SigmoidOp;
template <typename Backend> class EluOp;
template <typename Backend> class SeluOp;
template <typename Backend> class TanhOp;
template <typename Backend> class HardTanhOp;

class ReferenceBackend {

#define DECL_CREATE_OP(OpType)                                                 \
    static smaug::OpType<ReferenceBackend>* create##OpType(                    \
            const std::string& name, Workspace* workspace)

   public:
    static const int Alignment = 0;
    static const bool PrecomputeBNVariance = true;
    static const bool ColumnMajorFCWeights = true;
    static const std::string Name;

    DECL_CREATE_OP(ConvolutionOp);
    DECL_CREATE_OP(DataOp);
    DECL_CREATE_OP(DepthwiseConvolutionOp);
    DECL_CREATE_OP(MaxPoolingOp);
    DECL_CREATE_OP(AvgPoolingOp);
    DECL_CREATE_OP(InnerProductOp);
    DECL_CREATE_OP(SoftmaxOp);
    DECL_CREATE_OP(FlattenOp);
    DECL_CREATE_OP(BatchNormOp);
    DECL_CREATE_OP(EltwiseAddOp);
    DECL_CREATE_OP(ReluOp);
    DECL_CREATE_OP(SigmoidOp);
    DECL_CREATE_OP(EluOp);
    DECL_CREATE_OP(SeluOp);
    DECL_CREATE_OP(TanhOp);
    DECL_CREATE_OP(HardTanhOp);

#undef DECL_CREATE_OP

};

namespace smv {
extern int kSpadSize;
}  // namespace smv

class SmvConvolutionOp;
class SmvBackend {

#define DECL_CREATE_OP(OpType)                                                 \
    static smaug::OpType<SmvBackend>* create##OpType(                          \
            const std::string& name, Workspace* workspace)
#define DECL_CREATE_SMV_OP(OpType)                                             \
    static smaug::Smv##OpType* create##OpType(                                 \
            const std::string& name, Workspace* workspace)

   public:
    static const int Alignment = 8;
    static const bool PrecomputeBNVariance = true;
    static const bool ColumnMajorFCWeights = true;
    static const std::string Name;

    static int SpadSize() { return smv::kSpadSize; }

    DECL_CREATE_SMV_OP(ConvolutionOp);
    DECL_CREATE_OP(DataOp);
    DECL_CREATE_OP(DepthwiseConvolutionOp);
    DECL_CREATE_OP(MaxPoolingOp);
    DECL_CREATE_OP(AvgPoolingOp);
    DECL_CREATE_OP(InnerProductOp);
    DECL_CREATE_OP(SoftmaxOp);
    DECL_CREATE_OP(FlattenOp);
    DECL_CREATE_OP(BatchNormOp);
    DECL_CREATE_OP(EltwiseAddOp);
    DECL_CREATE_OP(ReluOp);
    DECL_CREATE_OP(SigmoidOp);
    DECL_CREATE_OP(EluOp);
    DECL_CREATE_OP(SeluOp);
    DECL_CREATE_OP(TanhOp);
    DECL_CREATE_OP(HardTanhOp);

#undef DECL_SMV_OP
#undef DECL_CREATE_OP

};

}  // namespace smaug

#endif
