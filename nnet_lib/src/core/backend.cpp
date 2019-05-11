#include "backend.h"
#include "operators/batch_norm_op.h"
#include "operators/convolution_op.h"
#include "operators/data_op.h"
#include "operators/depthwise_convolution_op.h"
#include "operators/eltwise_add_op.h"
#include "operators/elu_op.h"
#include "operators/inner_product_op.h"
#include "operators/pooling_op.h"
#include "operators/relu_op.h"
#include "operators/reorder_op.h"
#include "operators/sigmoid_op.h"
#include "operators/softmax_op.h"
#include "operators/tanh_op.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_inner_product_op.h"
#include "operators/smv/smv_pooling_op.h"

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
DEF_CREATE_OP(FlattenOp, ReferenceBackend)
DEF_CREATE_OP(BatchNormOp, ReferenceBackend)
DEF_CREATE_OP(EltwiseAddOp, ReferenceBackend)
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
DEF_CREATE_OP(DataOp, SmvBackend)
DEF_CREATE_OP(DepthwiseConvolutionOp, SmvBackend)
DEF_CREATE_OP(SoftmaxOp, SmvBackend)
DEF_CREATE_OP(ReorderOp, SmvBackend)
DEF_CREATE_OP(FlattenOp, SmvBackend)
DEF_CREATE_OP(BatchNormOp, SmvBackend)
DEF_CREATE_OP(EltwiseAddOp, SmvBackend)
DEF_CREATE_OP(ReluOp, SmvBackend)
DEF_CREATE_OP(SigmoidOp, SmvBackend)
DEF_CREATE_OP(EluOp, SmvBackend)
DEF_CREATE_OP(SeluOp, SmvBackend)
DEF_CREATE_OP(TanhOp, SmvBackend)
DEF_CREATE_OP(HardTanhOp, SmvBackend)

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
float* spad0;
float* spad1;
float* spad2;
}  // namespace smv


}  // namespace smaug
