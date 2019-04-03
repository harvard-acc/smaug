#ifndef _CORE_TENSOR_H_
#define _CORE_TENSOR_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>

#include <google/protobuf/repeated_field.h>

#include "core/datatypes.h"
#include "core/tensor.pb.h"
#include "utility/utils.h"

namespace smaug {

class TensorShape {
   public:
    TensorShape() : layout(DataLayout::UnknownLayout) {}
    TensorShape(std::vector<int> _dims, DataLayout _layout, int _alignment = 0)
            : dims_(_dims), padding_(dims_.size()), layout(_layout),
              alignment(_alignment) {
        computePadding();
    }
    TensorShape(const TensorShape& shape)
            : dims_(shape.dims_), padding_(shape.padding_),
              layout(shape.layout), alignment(shape.alignment) {}
    TensorShape(const TensorShapeProto& shapeProto) {
        std::copy(shapeProto.dims().begin(),
                  shapeProto.dims().end(),
                  std::back_inserter(dims_));
        padding_.resize(shapeProto.dims_size());
        layout = shapeProto.layout();
        alignment = shapeProto.alignment();
        computePadding();
    }

    const std::vector<int>& dims() const { return dims_; }
    const std::vector<int>& padding() const { return padding_; }
    int operator[](int index) const { return dims_[getIndex(index)]; }
    int& operator[](int index) { return dims_[getIndex(index)]; }
    int getStorageDim(int index) const {
        return dims_[getIndex(index)] + padding_[getIndex(index)];
    }
    bool operator==(const TensorShape& other) const {
        return (dims_ == other.dims_ && layout == other.layout);
    }
    DataLayout getLayout() const { return layout; }
    int ndims() const { return dims_.size(); }
    int size() const { return product(dims_); }
    int storageSize() const { return product(sum(dims_, padding_)); }
    int getAlignment() const { return alignment; }
    int getPadding(int index) const { return padding_[index]; }

   protected:
    int getIndex(int index) const {
        if (index >= 0) return index;
        return (dims_.size() + index);
    }

    void computePadding() {
        int ndims = dims_.size();
        padding_[ndims - 1] = calc_padding(dims_[ndims - 1], alignment);
        for (int i = 0; i < ndims - 1; i++)
            padding_[i] = 0;
    }
    std::vector<int> dims_;
    std::vector<int> padding_;
    DataLayout layout;
    int alignment;
};

// An iterator over a multidimensional tensor's indices, accounting for data
// alignment padding.
//
// The iterator tracks the current location as a coordinate and outputs the
// linearized index so that the data in a tensor can be accessed. While most
// commonly used to iterate through the contents of a tensor one by one, it can
// also provide random access to any location in the tensor.
//
// Example usage for simple iteration:
//   auto iter = TensorIndexIterator(tensor->getShape());
//   // OR: auto iter = tensor->startIndex();
//   float* data = tensor->data<float>();
//   while (!iter.end())
//      std::cout << data[iter] << ",";
//
// Example usage for random access (assume 4D tensor):
//   auto iter = TensorIndexIterator(tensor->getShape());
//   float* data = tensor->data<float>();
//   data[iter(1,2,3,4)] = 1.2;
//   data[iter(3,4,0,0)] = 3.4;
//
// The iterator skips over data alignment padding areas, if any exist.
class TensorIndexIterator {
   public:
    TensorIndexIterator(const TensorShape& shape, bool _atEnd = false)
            : dims(shape.dims()), padding(shape.padding()), atEnd(_atEnd) {
        state.resize(dims.size(), 0);
    }

    operator int() const { return getIndex(state); }

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

    template <typename... Args>
    int operator()(int i, Args... args) {
        auto indices = variadicToArray(i, args...);
        return getIndex(indices);
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
    template <typename Container>
    int getIndex(Container indices) const {
        int linearIndex = 0, stride = 1;
        for (int i = (int)indices.size() - 1; i >= 0; i--) {
            linearIndex += indices[i] * stride;
            stride *= (dims.at(i) + padding.at(i));
        }
        return linearIndex;
    }

    std::vector<int> state;
    std::vector<int> dims;
    std::vector<int> padding;
    bool atEnd;
};

// A tensor index iterator that stays within a specified rectangular region.
//
// The rectangular region is specified using an origin coordinate and a region
// size. The iterator will output linear indices in the same space as the full
// tensor index iterator, but indices outside the region will be skipped.
//
// Example: consider a 3x3 tensor. The upper right 2x2 region's origin is at
// location (0,1). We can output just that block like so:
//
//    auto it = TensorRegionIndexIterator(tensor->getShape(), {0,1}, {2,2});
//    while (!it.end())
//       std::cout << (int)it << "\n";
//
//  This produces: 1, 2, 4, 5
class TensorRegionIndexIterator : public TensorIndexIterator {
   public:
    TensorRegionIndexIterator(const TensorShape& shape,
                              const std::vector<int>& _origin,
                              const std::vector<int>& _regionSize)
            : TensorIndexIterator(shape, false), origin(_origin),
              regionSize(_regionSize) {
        state = origin;
    }

    void operator++() {
        bool carry = true;
        for (int i = (int)state.size() - 1; i >= 0 && carry; i--) {
            int currValue = state[i];
            currValue++;
            carry = (currValue >= dims[i] ||
                     currValue >= origin[i] + regionSize[i]);
            if (carry)
                currValue = origin[i];
            state[i] = currValue;
        }
        if (carry)
            atEnd = true;
    }

   protected:
    std::vector<int> origin;
    std::vector<int> regionSize;
};

std::ostream& operator<<(std::ostream& os, const TensorIndexIterator& iter);
std::ostream& operator<<(std::ostream& os, const TensorShape& shape);

class TensorBase {
   public:
    TensorBase() : name(""), dataFormat(UnknownStorageFormat) {}
    virtual ~TensorBase() {}

    // We could use this constructor for placeholder variables that don't have
    // any dynamic memory allocated yet.
    TensorBase(const std::string& _name, const TensorShape& _shape)
            : name(_name), shape(_shape), dataFormat(Uncompressed),
              dataType(UnknownDataType) {}

    TensorBase(const TensorProto& tensorProto)
            : name(tensorProto.name()), shape(tensorProto.shape()),
              dataFormat(tensorProto.data_format()),
              dataType(tensorProto.data_type()) {}

    // TODO: Do we need a copy constructor?

    std::string getName() const { return name; }
    const TensorShape& getShape() const { return shape; }
    int ndims() const { return shape.ndims(); }
    int dim(int index) const { return shape[index]; }
    int getTotalDim(int index) const { return shape.getStorageDim(index); }
    int getDataStorageFormat() const { return dataFormat; }
    DataType getDataType() const { return dataType; }
    int getDataTypeSize() const {
        switch (dataType) {
            case Float16:
                return 2;
            case Int32:
            case Float32:
                return 4;
            case Int64:
            case Float64:
                return 8;
            default:
                assert(false && "UnknownDataType has no size!");
        }
    }
    virtual bool containsData() const = 0;

   protected:
    std::string name;
    TensorShape shape;
    DataStorageFormat dataFormat;
    DataType dataType;
};

class Tensor : public TensorBase {
   public:
    Tensor() : TensorBase(), tensorData(NULL) {}
    Tensor(const std::string& _name, const TensorShape& _shape)
            : TensorBase(_name, _shape), tensorData(NULL) {}
    virtual ~Tensor() {}

    Tensor(const TensorProto& tensorProto)
            : TensorBase(tensorProto), tensorData(NULL) {
        DataType dataType = tensorProto.data_type();
        switch (dataType) {
            case Float16:
                fillHalfData(tensorProto.half_data());
                break;
            case Float32:
                fillData<float>(tensorProto.float_data());
                break;
            case Float64:
                fillData<double>(tensorProto.double_data());
                break;
            case Int32:
                fillData<int>(tensorProto.int_data());
                break;
            case Int64:
                fillData<int64_t>(tensorProto.int64_data());
            default:
                assert(false && "Unknown data format!");
        }
    }

    TensorIndexIterator startIndex() const {
        return TensorIndexIterator(shape);
    }

    virtual bool containsData() const { return tensorData != nullptr; }

    template <typename T>
    void fillData(T* externalData, int size) {
        T* rawPtr = data<T>();
        for (int i = 0; i < size; i++) {
            rawPtr[i] = externalData[i];
        }
    }

    template <typename T>
    void fillData(std::initializer_list<T> externalData) {
        T* rawPtr = data<T>();
        int i = 0;
        for (auto dataPtr = externalData.begin(); dataPtr != externalData.end();
             ++dataPtr, ++i) {
            rawPtr[i] = *dataPtr;
        }
    }

    template <typename T>
    void fillData(const google::protobuf::RepeatedField<T>& externalData) {
        allocateStorage<T>();
        T* rawPtr = data<T>();
        int i = 0;
        for (auto dataPtr = externalData.begin(); dataPtr != externalData.end();
             ++dataPtr, ++i) {
            rawPtr[i] = *dataPtr;
        }
    }

    // Fill the tensor with float16 data. This is needed because the data in
    // tensor proto packs two float16 into one int32. Here we do the unpacking.
    void fillHalfData(
            const google::protobuf::RepeatedField<int>& externalData) {
        allocateStorage<float16>();
        float16* rawPtr = data<float16>();
        for (int i = 0; i < shape.storageSize(); i++) {
            bool useLowHalf = (i % 2 == 0);
            rawPtr[i] = externalData[i / 2] >> (useLowHalf ? 0 : 16);
        }
    }

    template <typename T>
    T* allocateStorage() {
        if (tensorData == NULL) {
            dataType = ToDataType<T>::dataType;
            // TODO: Replace this with malloc_aligned.
            int size = shape.storageSize();
            assert(size > 0 && "Attempted to allocate zero storage!");
            tensorData = std::shared_ptr<void>(
                    new T[size], std::default_delete<T[]>());
        }
        return reinterpret_cast<T*>(tensorData.get());
    }

    void allocateStorage(DataType _dataType) {
        switch (_dataType) {
            case Float16:
                allocateStorage<float16>();
                return;
            case Float32:
                allocateStorage<float>();
                return;
            case Float64:
                allocateStorage<double>();
                return;
            case Int32:
                allocateStorage<int>();
                return;
            case Int64:
                allocateStorage<int64_t>();
                return;
            default:
                assert(false && "Unknown data type!");
        }
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

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

   protected:
    std::shared_ptr<void> tensorData;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
void copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      std::vector<int> destOrigin,
                      std::vector<int> srcOrigin,
                      std::vector<int> regionSize);

/* A container of tensors.
 *
 * Each of the tensors in a TiledTensor represents one tile of a large tensor.
 * Tensors can be accessed via an index iterator, such like scalar data in an
 * ordinary Tensor is accessed.
 */
class TiledTensor : public TensorBase {
  public:
    TiledTensor() : TensorBase() {}
    TiledTensor(const TensorShape& shape) : TensorBase("", shape) {
        tensors.resize(shape.size());
    }
    virtual bool containsData() const { return !tensors.empty(); }

    TensorIndexIterator startIndex() const {
        return TensorIndexIterator(shape);
    }

    const Tensor* operator[](int index) const {
        return tensors.at(index);
    }
    Tensor*& operator[](int index) { return tensors[index]; }
    int size() const { return shape.size(); }

    bool isDimNHTiled() const {
        if (tensors.empty())
            return false;
        if (shape.ndims() != 4)
            return false;
        // DimNH tiled means that there is more than one block in the row
        // dimension.
        return ((shape.getLayout() == DataLayout::NHWC && shape[1] > 1) ||
                (shape.getLayout() == DataLayout::NCHW && shape[2] > 1));
    }

   protected:
    std::vector<Tensor*> tensors;
};

// This will copy data from a tiled tensor into a single tensor. We name it as
// "untile" because what it does reverses the tiling process.
void untileTiledTensor(TiledTensor& tiledTensor, Tensor* destTensor);

template <typename DType>
void writeTensorToOstream(std::ostream& os, const Tensor& tensor) {
    const TensorShape& shape = tensor.getShape();
    if (shape.ndims() == 0) {
        os << "  [ ]\n";
        return;
    }
    int ndims = shape.ndims();
    int newlineAfterElems = shape[ndims - 1];
    int newGroupAfterElems =
            (shape.ndims() >= 2 ? shape[ndims - 1] * shape[ndims - 2]
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

namespace internal {
template <typename DType>
void copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      const std::vector<int>& destOrigin,
                      const std::vector<int>& srcOrigin,
                      const std::vector<int>& regionSize) {
    auto destIt =
            TensorRegionIndexIterator(dest->getShape(), destOrigin, regionSize);
    auto srcIt =
            TensorRegionIndexIterator(src->getShape(), srcOrigin, regionSize);
    DType* destPtr = dest->template data<DType>();
    DType* srcPtr = src->template data<DType>();
    while (!srcIt.end() && !destIt.end()) {
        destPtr[destIt] = srcPtr[srcIt];
        ++destIt;
        ++srcIt;
    }
}
}  // namespace internal

}  // namespace smaug

#endif
