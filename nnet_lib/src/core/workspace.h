#ifndef _CORE_WORKSPACE_H_
#define _CORE_WORKSPACE_H_

#include <map>
#include <string>

#include "core/tensor.h"

namespace smaug {

class Workspace {
  public:
    Workspace() {}

    template <typename Backend>
    void addTensor(Tensor<Backend>* tensor) {
        tensors[tensor->getName()] = static_cast<TensorBase*>(tensor);
    }

    template <typename Backend>
    Tensor<Backend>* getTensor(const std::string& name) const {
        if (tensors.find(name) == tensors.end())
            return nullptr;
        return dynamic_cast<Tensor<Backend>*>(tensors.at(name));
    }

    template <typename Backend>
    Tensor<Backend>* getTensor(Operator* op) const {
        return getTensor<Backend>(op->getName());
    }

   protected:
    std::map<std::string, TensorBase*> tensors;
};

}

#endif
