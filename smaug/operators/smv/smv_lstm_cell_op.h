#ifndef _OPERATORS_SMV_SMV_LSTM_OP_H_
#define _OPERATORS_SMV_SMV_LSTM_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/lstm_op.h"

namespace smaug {

class SmvLSTMCellOp : public LSTMCellOp<SmvBackend> {
  public:
    using LSTMCellOp<SmvBackend>::LSTMCellOp;
    void tile() override;
    void run() override;
};

}  // namespace smaug

#endif
