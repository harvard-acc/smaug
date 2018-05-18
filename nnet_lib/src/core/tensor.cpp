#include <iostream>

#include "core/tensor.h"

namespace smaug {

std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
    os << "(";
    for (int i = 0; i < shape.size(); i++) {
        os << shape[i];
        if (i != shape.size() - 1)
            os << ", ";
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const TensorIndexIterator& iter) {
    os << "( ";
    for (int i = 0; i < iter.dims.size(); ++i) {
        os << iter.state[i] << " ";
    }
    os << ")";
    return os;
}

}  // namespace smaug
