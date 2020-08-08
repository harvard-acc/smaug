#ifndef _CORE_WORKSPACE_H_
#define _CORE_WORKSPACE_H_

#include <map>
#include <string>

#include "smaug/core/tensor.h"
#include "smaug/core/operator.h"

namespace smaug {

 /**
  * Workspace is the container and owner of all Tensors and Operators in the
  * Network. Every Tensor/Operator that is created must be added to a
  * Workspace (and in general, there is only one Workspace).
  */
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

    void addTiledTensor(TiledTensor& tiledTensor) {
        for (auto i = tiledTensor.startIndex(); !i.end(); ++i) {
            Tensor* tensor = tiledTensor[i];
            tensors[tensor->getName()] = static_cast<TensorBase*>(tensor);
        }
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
