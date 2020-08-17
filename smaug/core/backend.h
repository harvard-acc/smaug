#ifndef _CORE_BACKEND_H_
#define _CORE_BACKEND_H_

#include <string>

#include "smaug/core/datatypes.h"
#include "smaug/utility/utils.h"

// These are compile-time switches that selectively build a copy of SMAUG with
// a particular backend.
#define REFERENCE 0
#define SMVBACKEND 1

/**
 * The smaug namespace is the parent namespace of all C++ code in SMAUG.
 */
namespace smaug {

/**
 * The list of all hardware backends in the system.
 */
enum BackendName {
    /** Reference backend. */
    Reference = REFERENCE,
    /** SMV backend. */
    Smv = SMVBACKEND,
    /** Invalid backend. */
    UnknownBackend,
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class Workspace;
template <typename Backend> class ConvolutionOp;
template <typename Backend> class DataOp;
template <typename Backend> class DepthwiseConvolutionOp;
template <typename Backend> class MaxPoolingOp;
template <typename Backend> class AvgPoolingOp;
template <typename Backend> class InnerProductOp;
template <typename Backend> class SoftmaxOp;
template <typename Backend> class ReorderOp;
template <typename Backend> class ConcatOp;
template <typename Backend> class SplitOp;
template <typename Backend> class ReshapeOp;
template <typename Backend> class RepeatOp;
template <typename Backend> class FlattenOp;
template <typename Backend> class BatchNormOp;
template <typename Backend> class EltwiseAddOp;
template <typename Backend> class EltwiseMulOp;
template <typename Backend> class LessOp;
template <typename Backend> class LessEqualOp;
template <typename Backend> class GreaterOp;
template <typename Backend> class GreaterEqualOp;
template <typename Backend> class SwitchOp;
template <typename Backend> class MergeOp;
template <typename Backend> class ReluOp;
template <typename Backend> class SigmoidOp;
template <typename Backend> class EluOp;
template <typename Backend> class SeluOp;
template <typename Backend> class TanhOp;
template <typename Backend> class HardTanhOp;
#endif

/**
 * The ref namespace contains all code specific to the Reference backend.
 */
namespace ref {
extern const unsigned kConvolutionHw;
extern const unsigned kInnerProductHw;
extern const unsigned kEltwiseOpHw;
extern const unsigned kBatchNormHw;
extern const unsigned kPoolingHw;
}  // namespace ref

/**
 * ReferenceBackend provides reference implementations of all operators
 * supported by SMAUG. Some of these operators are written to be
 * Aladdin-compatible for simulation purposes; others are intended to be
 * executed in software. As the name suggests, these implementations
 * are not optimized in any way.
 */
class ReferenceBackend {

#define DECL_CREATE_OP(OpType)                                                 \
    static smaug::OpType<ReferenceBackend>* create##OpType(                    \
            const std::string& name, Workspace* workspace)

   public:
    static const int Alignment = 0;
    static const bool PrecomputeBNVariance = true;
    static const bool TransposeFCWeights = false;
    static const std::string Name;
    static const DataLayout DefaultInputDataLayout = DataLayout::NCHW;

    DECL_CREATE_OP(ConvolutionOp);
    DECL_CREATE_OP(DataOp);
    DECL_CREATE_OP(DepthwiseConvolutionOp);
    DECL_CREATE_OP(MaxPoolingOp);
    DECL_CREATE_OP(AvgPoolingOp);
    DECL_CREATE_OP(InnerProductOp);
    DECL_CREATE_OP(SoftmaxOp);
    DECL_CREATE_OP(ReorderOp);
    DECL_CREATE_OP(ConcatOp);
    DECL_CREATE_OP(SplitOp);
    DECL_CREATE_OP(ReshapeOp);
    DECL_CREATE_OP(RepeatOp);
    DECL_CREATE_OP(FlattenOp);
    DECL_CREATE_OP(BatchNormOp);
    DECL_CREATE_OP(EltwiseAddOp);
    DECL_CREATE_OP(EltwiseMulOp);
    DECL_CREATE_OP(LessOp);
    DECL_CREATE_OP(LessEqualOp);
    DECL_CREATE_OP(GreaterOp);
    DECL_CREATE_OP(GreaterEqualOp);
    DECL_CREATE_OP(SwitchOp);
    DECL_CREATE_OP(MergeOp);
    DECL_CREATE_OP(ReluOp);
    DECL_CREATE_OP(SigmoidOp);
    DECL_CREATE_OP(EluOp);
    DECL_CREATE_OP(SeluOp);
    DECL_CREATE_OP(TanhOp);
    DECL_CREATE_OP(HardTanhOp);

#undef DECL_CREATE_OP

};

/**
 * The smv namespace contains all code specific to the Smv backend.
 */
namespace smv {
extern int kSpadSize;
extern const unsigned kConvolutionHw;
extern const unsigned kInnerProductHw;
extern const unsigned kEltwiseOpHw;
extern const unsigned kBatchNormHw;
extern const unsigned kPoolingHw;
extern const unsigned kSystolicArrayHw;
// Note that these naked pointers are never to be used except when invoking the
// kernels themselves.
extern float* spad0;
extern float* spad1;
extern float* spad2;
}  // namespace smv

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class SmvConvolutionOp;
class SmvInnerProductOp;
class SmvMaxPoolingOp;
class SmvAvgPoolingOp;
class SmvBatchNormOp;
class SmvReluOp;
class SmvEluOp;
class SmvSeluOp;
class SmvTanhOp;
class SmvHardTanhOp;
class SmvSigmoidOp;
class SmvSoftmaxOp;
class SmvEltwiseAddOp;
class SmvEltwiseMulOp;
class SmvLessOp;
class SmvLessEqualOp;
class SmvGreaterOp;
class SmvGreaterEqualOp;
#endif

/**
 * SmvBackend implements a set of models of optimized DL kernels that were
 * taped out on a machine learning SoC by the Harvard Architecture, Circuits,
 * and Compilers. All operators implemented in this backend are vectorized
 * using eight fp16 values. See the individual operators for more details.
 */
class SmvBackend {

// The difference between DECL_CREATE_OP and DECL_CREATE_SMV_OP is the latter is
// used for a custom operator implementation for the SMV backend, which is not
// limited to explicit template specializations.
#define DECL_CREATE_OP(OpType)                                                 \
    static smaug::OpType<SmvBackend>* create##OpType(                          \
            const std::string& name, Workspace* workspace)
#define DECL_CREATE_SMV_OP(OpType)                                             \
    static smaug::Smv##OpType* create##OpType(                                 \
            const std::string& name, Workspace* workspace)

   public:
    static const int Alignment = 8;
    static const bool PrecomputeBNVariance = true;
    static const bool TransposeFCWeights = true;
    static const std::string Name;
    static const DataLayout DefaultInputDataLayout = DataLayout::NHWC;

    static int SpadSize() { return smv::kSpadSize; }
    static void initGlobals() {
        // kSpadSize is in terms of float16 data.
        smv::kSpadSize = 32 * 1024;
        // In SMV, all tensors store float16 data, but due to the modelling
        // restriction of Aladdin, we actually store float32 data in the
        // scratchpads. This why the allocated memory size here is double
        // kSpadSize.
        smv::spad0 = (float*)malloc_aligned(smv::kSpadSize * 2);
        smv::spad1 = (float*)malloc_aligned(smv::kSpadSize * 2);
        smv::spad2 = (float*)malloc_aligned(smv::kSpadSize * 2);
    }
    static void freeGlobals() {
        free(smv::spad0);
        free(smv::spad1);
        free(smv::spad2);
    }

    DECL_CREATE_SMV_OP(ConvolutionOp);
    DECL_CREATE_SMV_OP(InnerProductOp);
    DECL_CREATE_SMV_OP(MaxPoolingOp);
    DECL_CREATE_SMV_OP(AvgPoolingOp);
    DECL_CREATE_SMV_OP(BatchNormOp);
    DECL_CREATE_SMV_OP(ReluOp);
    DECL_CREATE_SMV_OP(EluOp);
    DECL_CREATE_SMV_OP(SeluOp);
    DECL_CREATE_SMV_OP(TanhOp);
    DECL_CREATE_SMV_OP(HardTanhOp);
    DECL_CREATE_SMV_OP(SigmoidOp);
    DECL_CREATE_SMV_OP(SoftmaxOp);
    DECL_CREATE_SMV_OP(EltwiseAddOp);
    DECL_CREATE_SMV_OP(EltwiseMulOp);
    DECL_CREATE_SMV_OP(LessOp);
    DECL_CREATE_SMV_OP(LessEqualOp);
    DECL_CREATE_SMV_OP(GreaterOp);
    DECL_CREATE_SMV_OP(GreaterEqualOp);
    DECL_CREATE_OP(DataOp);
    DECL_CREATE_OP(DepthwiseConvolutionOp);
    DECL_CREATE_OP(ReorderOp);
    DECL_CREATE_OP(ConcatOp);
    DECL_CREATE_OP(SplitOp);
    DECL_CREATE_OP(ReshapeOp);
    DECL_CREATE_OP(RepeatOp);
    DECL_CREATE_OP(FlattenOp);
    DECL_CREATE_OP(SwitchOp);
    DECL_CREATE_OP(MergeOp);

#undef DECL_SMV_OP
#undef DECL_CREATE_OP

};

}  // namespace smaug

#endif
