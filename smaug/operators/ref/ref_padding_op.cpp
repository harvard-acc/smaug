#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/padding_op.h"

namespace smaug {

template <>
void PaddingOp<ReferenceBackend>::run() {
    ;
}

}  // namespace smaug