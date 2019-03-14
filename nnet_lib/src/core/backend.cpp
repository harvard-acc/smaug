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
DEF_CREATE_OP(DataOp, SmvBackend)
DEF_CREATE_OP(DepthwiseConvolutionOp, SmvBackend)
DEF_CREATE_OP(MaxPoolingOp, SmvBackend)
DEF_CREATE_OP(AvgPoolingOp, SmvBackend)
DEF_CREATE_OP(InnerProductOp, SmvBackend)
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

namespace smv {
int kUmemSize;
int kSpadSize;
}  // namespace smv


}  // namespace smaug
