#ifndef _CORE_WORKSPACE_H_
#define _CORE_WORKSPACE_H_

#include <map>
#include <string>

#include "core/tensor.h"

namespace smaug {

class Workspace {
  public:
    Workspace() {}
    ~Workspace() {
        for (auto& tensor : tensors)
            delete tensor.second;
    }

    Tensor* addTensor(Tensor* tensor) {
        tensors[tensor->getName()] = static_cast<TensorBase*>(tensor);
        return tensor;
    }

    Tensor* getTensor(const std::string& name) const {
        if (tensors.find(name) == tensors.end())
            return nullptr;
        return dynamic_cast<Tensor*>(tensors.at(name));
    }

    Tensor* getTensor(Operator* op) const {
        return getTensor(op->getName());
    }

   protected:
    std::map<std::string, TensorBase*> tensors;
};

}

#endif
