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

}  // namespace smaug
