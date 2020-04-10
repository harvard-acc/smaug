#include "backend.h"
#include "operators/batch_norm_op.h"
#include "operators/convolution_op.h"
#include "operators/data_op.h"
#include "operators/depthwise_convolution_op.h"
#include "operators/eltwise_add_op.h"
#include "operators/eltwise_mul_op.h"
#include "operators/elu_op.h"
#include "operators/inner_product_op.h"
#include "operators/pooling_op.h"
#include "operators/relu_op.h"
#include "operators/reorder_op.h"
#include "operators/concat_op.h"
#include "operators/split_op.h"
#include "operators/sigmoid_op.h"
#include "operators/softmax_op.h"
#include "operators/tanh_op.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_inner_product_op.h"
#include "operators/smv/smv_pooling_op.h"
#include "operators/smv/smv_batch_norm_op.h"
#include "operators/smv/smv_relu_op.h"
#include "operators/smv/smv_elu_op.h"
#include "operators/smv/smv_tanh_op.h"
#include "operators/smv/smv_sigmoid_op.h"
#include "operators/smv/smv_eltwise_add_op.h"
#include "operators/smv/smv_eltwise_mul_op.h"

namespace smaug {

#define DEF_CREATE_OP(OpType, Backend)                                         \
    OpType<Backend>* Backend::create##OpType(                                  \
            const std::string& name, Workspace* workspace) {                   \
        return new OpType<Backend>(name, workspace);                           \
    }

#define DEF_CREATE_SMV_OP(OpType)                                              \
    Smv##OpType* SmvBackend::create##OpType(                                   \
            const std::string& name, Workspace* workspace) {                   \
        return new Smv##OpType(name, workspace);                               \
    }

const std::string ReferenceBackend::Name = "Reference";
const std::string SmvBackend::Name = "SMV";

DEF_CREATE_OP(ConvolutionOp, ReferenceBackend)
DEF_CREATE_OP(DataOp, ReferenceBackend)
DEF_CREATE_OP(DepthwiseConvolutionOp, ReferenceBackend)
DEF_CREATE_OP(MaxPoolingOp, ReferenceBackend)
DEF_CREATE_OP(AvgPoolingOp, ReferenceBackend)
DEF_CREATE_OP(InnerProductOp, ReferenceBackend)
DEF_CREATE_OP(SoftmaxOp, ReferenceBackend)
DEF_CREATE_OP(ReorderOp, ReferenceBackend)
DEF_CREATE_OP(ConcatOp, ReferenceBackend)
DEF_CREATE_OP(SplitOp, ReferenceBackend)
DEF_CREATE_OP(FlattenOp, ReferenceBackend)
DEF_CREATE_OP(BatchNormOp, ReferenceBackend)
DEF_CREATE_OP(EltwiseAddOp, ReferenceBackend)
DEF_CREATE_OP(EltwiseMulOp, ReferenceBackend)
DEF_CREATE_OP(ReluOp, ReferenceBackend)
DEF_CREATE_OP(SigmoidOp, ReferenceBackend)
DEF_CREATE_OP(EluOp, ReferenceBackend)
DEF_CREATE_OP(SeluOp, ReferenceBackend)
DEF_CREATE_OP(TanhOp, ReferenceBackend)
DEF_CREATE_OP(HardTanhOp, ReferenceBackend)

DEF_CREATE_SMV_OP(ConvolutionOp)
DEF_CREATE_SMV_OP(InnerProductOp)
DEF_CREATE_SMV_OP(MaxPoolingOp)
DEF_CREATE_SMV_OP(AvgPoolingOp)
DEF_CREATE_SMV_OP(BatchNormOp)
DEF_CREATE_SMV_OP(ReluOp)
DEF_CREATE_SMV_OP(EluOp)
DEF_CREATE_SMV_OP(SeluOp)
DEF_CREATE_SMV_OP(TanhOp)
DEF_CREATE_SMV_OP(HardTanhOp)
DEF_CREATE_SMV_OP(SigmoidOp)
DEF_CREATE_SMV_OP(EltwiseAddOp)
DEF_CREATE_SMV_OP(EltwiseMulOp)
DEF_CREATE_OP(DataOp, SmvBackend)
DEF_CREATE_OP(DepthwiseConvolutionOp, SmvBackend)
DEF_CREATE_OP(SoftmaxOp, SmvBackend)
DEF_CREATE_OP(ReorderOp, SmvBackend)
DEF_CREATE_OP(ConcatOp, SmvBackend)
DEF_CREATE_OP(SplitOp, SmvBackend)
DEF_CREATE_OP(FlattenOp, SmvBackend)

namespace ref {
const unsigned kConvolutionHw = 0x0001;
const unsigned kInnerProductHw = 0x0002;
const unsigned kEltwiseOpHw = 0x0003;
const unsigned kBatchNormHw = 0x0004;
const unsigned kPoolingHw = 0x0005;
}  // namespace ref

namespace smv {
int kSpadSize;
// Use the same accelerator id for all hardware blocks. This means we will
// simulate only ONE datapath instead of multiple, which means that the two
// blocks can share the scratchpads (without any infrastructure
// changes). The key is that we still trace the functions at the _hw level,
// so that Aladdin will exit after simulating each block, and we can return
// control to the CPU at the right places.  In contrast, if we used two
// different ids, we would have two different datapaths that could not share
// data directly.
const unsigned kConvolutionHw = 0x0003;
const unsigned kInnerProductHw = 0x0003;
const unsigned kEltwiseOpHw = 0x0003;
const unsigned kBatchNormHw = 0x0003;
const unsigned kPoolingHw = 0x0003;
// The systolic array is implemented in gem5 instead of Aladdin, so it needs to
// have a different accelerator id.
const unsigned kSystolicArrayHw = 0x0004;
float* spad0;
float* spad1;
float* spad2;
}  // namespace smv


}  // namespace smaug
