#ifndef _CORE_TENSOR_H_
#define _CORE_TENSOR_H_

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "core/datatypes.h"
#include "utility/utils.h"

namespace smaug {

class TensorShape {
   public:
    TensorShape() : layout(DataLayout::UnknownLayout) {}
    TensorShape(std::vector<int> _dims, DataLayout _layout)
            : dims_(_dims), layout(_layout) {}

    const std::vector<int>& dims() const { return dims_; }
    int& operator[](int index) { return dims_[index]; }
    int operator[](int index) const { return dims_[index]; }
    bool operator==(const TensorShape& other) const {
        return (dims_ == other.dims_ && layout == other.layout);
    }
    DataLayout getLayout() const { return layout; }
    int size() const { return dims_.size(); }
    int total() const { return product(dims_); }

   protected:
    std::vector<int> dims_;
    DataLayout layout;
};

std::ostream& operator<<(std::ostream& os, const TensorShape& shape);

class TensorBase {
   public:
    TensorBase() : name(""), alignment(0), dataFormat(UnknownStorageFormat) {}

    // We could use this constructor for placeholder variables that don't have
    // any dynamic memory allocated yet.
    TensorBase(const std::string& _name,
               const TensorShape& _shape,
               int _alignment = 0)
            : name(_name), shape(_shape), dataFormat(Uncompressed),
              dataType(UnknownDataType), alignment(_alignment),
              padding(shape.size()) {
        assert(shape.size() == padding.size());
        computePadding();
    }

    virtual ~TensorBase() {}

    // TODO: Do we need a copy constructor?

    std::string getName() const { return name; }
    const TensorShape& getShape() const { return shape; }
    int ndims() const { return shape.size(); }
    int dim(int index) const { return shape[index]; }
    int getPadding(int index) const { return padding[index]; }
    int getTotalDim(int index) const {
        return shape[index] + padding[index];
    }
    int getAlignment() const { return alignment; }
    int getDataStorageFormat() const { return dataFormat; }

    void setAlignment(int align) { alignment = align; }

   protected:
    void computePadding() {
       padding[0] = calc_padding(shape[0], alignment);
       for (int i = 1; i < shape.size(); i++)
           padding[i] = 0;
    }

    std::string name;
    TensorShape shape;
    DataStorageFormat dataFormat;
    DataType dataType;
    int alignment;
    std::vector<int> padding;
};

template <typename Backend>
class Tensor : public TensorBase {
  public:
    Tensor() : TensorBase(), tensorData(NULL) {}
    Tensor(const std::string& _name, const TensorShape& _shape)
            : TensorBase(_name, _shape, Backend::Alignment), tensorData(NULL) {}

    template <typename T>
    Tensor(const std::string& _name,
           const TensorShape& _shape,
           const std::vector<T>& _data)
            : TensorBase(_name, _shape, Backend::Alignment) {
        assert(product(sum(shape.dims(), padding)) == _data.size());
        dataType = ToDataType<T>::type;
        allocateStorage<T>();
        copyFromExternalData(_data.data(), _data.size());
    }

    template <typename T>
    Tensor(const std::string& _name, const TensorShape& _shape, T* _data)
            : TensorBase(_name, _shape, Backend::Alignment) {
        dataType = ToDataType<T>::type;
        allocateStorage<T>();
        copyFromExternalData(_data, product(sum(shape.dims(), padding)));
    }

    virtual ~Tensor() {}

    template <typename T>
    void copyFromExternalData(T* externalData, int size) {
        T* rawPtr = data<T>();
        for (int i = 0; i < size; i++) {
            rawPtr[i] = externalData[i];
        }
    }

    template <typename T>
    void allocateStorage() {
        if (tensorData != NULL)
          return;
        // TODO: Replace this with malloc_aligned.
        int size = product(sum(shape.dims(), padding));
        tensorData = std::shared_ptr<void>(new T[size]);
    }

    template <typename T>
    const T* data() {
      assert(ToDataType<T>::dataType == dataType);
      return reinterpret_cast<T*>(tensorData.get());
    }

  protected:
    std::shared_ptr<void> tensorData;
};

}  // namespace smaug

#endif
