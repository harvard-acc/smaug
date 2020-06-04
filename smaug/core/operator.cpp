#include "smaug/core/operator.h"

namespace smaug {

bool Operator::isDead() {
    bool anyInputDead = false;
    for (auto input : inputs) {
        if (input->isDead()) {
            anyInputDead = true;
            break;
        }
    }
    return anyInputDead;
}

void Operator::printSummary(std::ostream& out) const {
    boost::format fmter(kLayerFormat);
    out << fmter % (this->name + " (" + OpType_Name(opType) + ")") %
                    outputs.at(0)->getShape() % getNumParameters();
    if (outputs.size() > 1) {
        for (int i = 1; i < outputs.size(); i++)
            out << fmter % "" % outputs.at(i)->getShape() % "";
    }
}

bool Operator::tensorsAllConstructed(
        const std::vector<TensorBase*>& tensors) const {
    for (auto tensor : tensors) {
        if (!tensor || !tensor->containsData())
            return false;
    }
    return true;
}

bool Operator::validateInputsOutputs() const {
    bool success = true;
    if (!tensorsAllConstructed(inputs)) {
        success = false;
        std::cerr << "[ERROR]: Inputs to " << getName()
                  << " were not all constructed!\n";
    }
    if (!tensorsAllConstructed(outputs)) {
        success = false;
        std::cerr << "[ERROR]: Outputs to " << getName()
                  << " were not all constructed!\n";
    }
    return success;
}

}  // namespace smaug
