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

class TensorIndexIterator {
   public:
    TensorIndexIterator(const std::vector<int>& _dims,
                        const std::vector<int>& _padding,
                        bool _atEnd = false)
            : dims(_dims), padding(_padding), atEnd(_atEnd) {
        state.resize(dims.size(), 0);
    }

    int getIndex() const {
        int linearIndex = 0, stride = 1;
        for (int i = (int)state.size() - 1; i >= 0; i--) {
            linearIndex += state[i] * stride;
            stride *= (dims.at(i) + padding.at(i));
        }
        return linearIndex;
    }

    operator int() const {
        return getIndex();
    }

    bool end() const { return atEnd; }

    void operator++() {
        bool carry = true;
        for (int i = (int)state.size() - 1; i >= 0 && carry; i--) {
            int currValue = state[i];
            currValue++;
            carry = (currValue >= dims[i]);
            if (carry)
                currValue = 0;
            state[i] = currValue;
        }
        if (carry)
            atEnd = true;
    }

    bool operator==(const TensorIndexIterator& other) const {
        return (state == other.state && dims == other.dims &&
                padding == other.padding && atEnd == other.atEnd);
    }

    bool operator!=(const TensorIndexIterator& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os,
                                    const TensorIndexIterator& iter);

   protected:
    std::vector<int> state;
    std::vector<int> dims;
    std::vector<int> padding;
    bool atEnd;
};

std::ostream& operator<<(std::ostream& os, const TensorIndexIterator& iter);

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
    DataType getDataType() const { return dataType; }
    virtual bool containsData() const = 0;

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
        allocateStorage<T>();
        copyFromExternalData(_data.data(), _data.size());
    }

    template <typename T>
    Tensor(const std::string& _name, const TensorShape& _shape, T* _data)
            : TensorBase(_name, _shape, Backend::Alignment) {
        allocateStorage<T>();
        copyFromExternalData(_data, product(sum(shape.dims(), padding)));
    }

    virtual ~Tensor() {}

    TensorIndexIterator startIndex() const {
        return TensorIndexIterator(shape.dims(), padding);
    }

    virtual bool containsData() const {
        return tensorData != nullptr;
    }

    template <typename T>
    void copyFromExternalData(T* externalData, int size) {
        T* rawPtr = data<T>();
        for (int i = 0; i < size; i++) {
            rawPtr[i] = externalData[i];
        }
    }

    template <typename T>
    T* allocateStorage() {
        if (tensorData == NULL) {
            dataType = ToDataType<T>::dataType;
            // TODO: Replace this with malloc_aligned.
            int size = product(sum(shape.dims(), padding));
            tensorData = std::shared_ptr<void>(new T[size]);
        }
        return reinterpret_cast<T*>(tensorData.get());
    }

    template <typename T>
    T* const data() const {
        assert(ToDataType<T>::dataType == dataType);
        return reinterpret_cast<T*>(tensorData.get());
    }

    template <typename T>
    T* data() {
        assert(ToDataType<T>::dataType == dataType);
        return reinterpret_cast<T*>(tensorData.get());
    }

    template <typename B>
    friend std::ostream& operator<<(std::ostream& os,
                                    const Tensor<B>& tensor);

   protected:
    std::shared_ptr<void> tensorData;
};

template <typename DType, typename Backend>
void writeTensorToOstream(std::ostream& os, const Tensor<Backend>& tensor) {
    const TensorShape& shape = tensor.template getShape();
    if (shape.size() == 0) {
        os << "  [ ]\n";
        return;
    }
    int ndims = shape.size();
    int newlineAfterElems = shape[ndims- 1];
    int newGroupAfterElems =
            (shape.size() >= 2 ? shape[ndims - 1] * shape[ndims - 2]
                               : shape[ndims - 1]);
    int counter = 0;
    DType* data = tensor.template data<DType>();
    os << tensor.getName() << ", shape = " << shape << "\n";
    for (auto idx = tensor.startIndex(); !idx.end(); ++idx) {
        // Print the current index after going through all of the last two
        // dimensions.
        if (counter == 0)
            os << idx << "\n[ ";
        os << data[idx] << " ";
        ++counter;
        if (counter % newGroupAfterElems == 0) {
            counter = 0;
            os << " ]\n";
        } else if (counter % newlineAfterElems == 0) {
            os << "\n  ";
        }
    }
}

template <typename Backend>
std::ostream& operator<<(std::ostream& os, const Tensor<Backend>& tensor) {
    DataType type = tensor.template getDataType();
    switch (type) {
      case Int32:
        writeTensorToOstream<int>(os, tensor);
        break;
      case Int64:
        writeTensorToOstream<int64_t>(os, tensor);
        break;
      case Float16:
        writeTensorToOstream<uint16_t>(os, tensor);
        break;
      case Float32:
        writeTensorToOstream<float>(os, tensor);
        break;
      case Float64:
        writeTensorToOstream<double>(os, tensor);
        break;
      default:
        assert(false && "Unknown data type!");
    }
    return os;
}

}  // namespace smaug

#endif
