#ifndef _CORE_TENSOR_H_
#define _CORE_TENSOR_H_

#include <cassert>
#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>

#include <google/protobuf/repeated_field.h>

#include "smaug/core/datatypes.h"
#include "smaug/core/tensor.pb.h"
#include "smaug/utility/utils.h"

namespace smaug {

class TensorShape {
   public:
    TensorShape() : layout(DataLayout::UnknownLayout) {}
    TensorShape(std::vector<int> _dims, DataLayout _layout, int _alignment = 0)
            : dims_(_dims), padding_(dims_.size()), layout(_layout),
              alignment(_alignment) {
        computePadding();
    }
    TensorShape(std::initializer_list<int> _dims,
                DataLayout _layout,
                int _alignment = 0)
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

    // Return a TensorShapeProto that serializes this TensorShape.
    TensorShapeProto* asTensorShapeProto();

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
            : dims(shape.dims()), padding(shape.padding()), atEnd(_atEnd),
              advanceOne(std::vector<int>(dims.size(), 1)) {
        state.resize(dims.size(), 0);
    }

    operator int() const { return getIndex(state); }

    bool end() const { return atEnd; }

    void operator++() { advanceRegion(advanceOne); }

    void operator+=(const std::vector<int>& region) {
        assert(region.size() == state.size());
        advanceRegion(region);
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

    // This returns the current index of the iterator on the specified
    // dimension.
    int currentIndex(int dim) const { return state[dim]; }

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

    virtual void advanceRegion(const std::vector<int>& region) {
        bool carry = true;
        for (int i = (int)state.size() - 1; i >= 0 && carry; i--) {
            int currValue = state[i] + region[i];
            carry = (currValue >= dims[i]);
            if (carry)
                currValue = 0;
            state[i] = currValue;
        }
        if (carry)
            atEnd = true;
    }

    std::vector<int> state;
    std::vector<int> dims;
    std::vector<int> padding;
    bool atEnd;
    const std::vector<int> advanceOne;
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

   protected:
    // Advance the tensor region index with the specified region size.
    virtual void advanceRegion(const std::vector<int>& advanceRegionSize) {
        bool carry = true;
        for (int i = (int)state.size() - 1; i >= 0 && carry; i--) {
            int currValue = state[i] + advanceRegionSize[i];
            carry = (currValue >= dims[i] ||
                     currValue >= origin[i] + regionSize[i]);
            if (carry)
                currValue = origin[i];
            state[i] = currValue;
        }
        if (carry)
            atEnd = true;
    }

    std::vector<int> origin;
    std::vector<int> regionSize;
};

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
                return sizeof(float16);
            case Int32:
                return sizeof(int32_t);
            case Float32:
                return sizeof(float);
            case Int64:
                return sizeof(int64_t);
            case Float64:
                return sizeof(double);
            case Bool:
                return sizeof(bool);
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

    Tensor(const TensorProto& tensorProto, const TensorData& tensorData)
            : TensorBase(tensorProto), tensorData(NULL) {
        DataType dataType = tensorProto.data_type();
        switch (dataType) {
            case Float16:
                fillHalfData(tensorData.half_data());
                break;
            case Float32:
                fillData<float>(tensorData.float_data());
                break;
            case Float64:
                fillData<double>(tensorData.double_data());
                break;
            case Int32:
                fillData<int>(tensorData.int_data());
                break;
            case Int64:
                fillData<int64_t>(tensorData.int64_data());
            case Bool:
                fillData<bool>(tensorData.bool_data());
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
#ifdef USE_PEDANTIC_COPY
        for (int i = 0; i < size; i++) {
            rawPtr[i] = externalData[i];
        }
#else
        std::copy(externalData, externalData + size, rawPtr);
#endif
    }

    template <typename T>
    void fillData(std::initializer_list<T> externalData) {
        T* rawPtr = data<T>();
#ifdef USE_PEDANTIC_COPY
        int i = 0;
        for (auto dataPtr = externalData.begin(); dataPtr != externalData.end();
             ++dataPtr, ++i) {
            rawPtr[i] = *dataPtr;
        }
#else
        std::copy(externalData.begin(), externalData.end(), rawPtr);
#endif
    }

    template <typename T>
    void fillData(const google::protobuf::RepeatedField<T>& externalData) {
        allocateStorage<T>();
        T* rawPtr = data<T>();
#ifdef USE_PEDANTIC_COPY
        int i = 0;
        for (auto dataPtr = externalData.begin(); dataPtr != externalData.end();
             ++dataPtr, ++i) {
            rawPtr[i] = *dataPtr;
        }
#else
        std::copy(externalData.begin(), externalData.end(), rawPtr);
#endif
    }

    // Fill the tensor with float16 data. This is needed because the data in
    // tensor proto packs two float16 into one int32.
    void fillHalfData(
            const google::protobuf::RepeatedField<int>& externalData) {
        allocateStorage<float16>();
        float16* rawPtr = data<float16>();
#ifdef USE_PEDANTIC_COPY
        for (int i = 0; i < shape.storageSize(); i++) {
            bool useLowHalf = (i % 2 == 0);
            rawPtr[i] = externalData[i / 2] >> (useLowHalf ? 0 : 16);
        }
#else
        const int* externalPtr = externalData.data();
        memcpy(rawPtr, externalPtr, shape.storageSize() * sizeof(float16));
#endif
    }

    template <typename T>
    T* allocateStorage() {
        if (tensorData == NULL) {
            dataType = ToDataType<T>::dataType;
            int size = shape.storageSize();
            assert(size > 0 && "Attempted to allocate zero storage!");
            tensorData = std::shared_ptr<void>(
                    malloc_aligned(size * sizeof(T), false), free);
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
            case Bool:
                allocateStorage<bool>();
                return;
            default:
                assert(false && "Unknown data type!");
        }
    }

    // Return a TensorProto that serializes this Tensor.
    TensorProto* asTensorProto();

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

/* A container of tensors.
 *
 * Each of the tensors in a TiledTensor represents one tile of a large tensor.
 * Tensors can be accessed via an index iterator, such like scalar data in an
 * ordinary Tensor is accessed.
 */
class TiledTensor : public TensorBase {
  public:
   TiledTensor(Tensor* _origTensor = nullptr, bool _useRawTensor = false)
           : TensorBase(), origTensor(_origTensor), useRawTensor(_useRawTensor),
             dataFilled(false) {}
   TiledTensor(const TensorShape& shape,
               Tensor* _origTensor = nullptr,
               bool _useRawTensor = false)
           : TensorBase("", shape), origTensor(_origTensor),
             useRawTensor(_useRawTensor), dataFilled(false) {
       tiles.resize(shape.size());
   }

   virtual bool containsData() const { return !tiles.empty(); }

   TensorIndexIterator startIndex() const { return TensorIndexIterator(shape); }

   const Tensor* operator[](int index) const { return tiles.at(index).tensor; }
   Tensor*& operator[](int index) { return tiles[index].tensor; }
   int size() const { return shape.size(); }

   bool isDimNHTiled() const {
       if (tiles.empty())
           return false;
       if (shape.ndims() != 4)
           return false;
       // DimNH tiled means that there is more than one block in the row
       // dimension.
       return ((shape.getLayout() == DataLayout::NHWC && shape[1] > 1) ||
               (shape.getLayout() == DataLayout::NCHW && shape[2] > 1));
   }

   // Return the tile's tensor with its data copied from the original tensor.
   Tensor* getTileWithData(int index);

   // Set the tile in the TiledTensor, with the option to copy data to it.
   void setTile(int index,
                const std::vector<int>& origin,
                Tensor* tensor,
                bool copyData);

   // Copy data (if needed) to all the tiles from the original tensor.
   void copyDataToAllTiles();

   // This will copy data from the tiled tensor into the original tensor. We
   // name it as "untile" because what it does reverses the tiling process.
   void untile();

   static void* tileCopyWorker(void* _args);

  protected:
   struct Tile {
       Tensor* tensor;
       // The tile's origins in the original tensor.
       std::vector<int> origin;
       // True if the tile has its origin set.
       bool hasOrigin;
       // True if we have copied data to this tile.
       bool hasData;

       Tile() : tensor(nullptr), origin(), hasOrigin(false), hasData(false) {}
   };

   // Scatter copies data from the tensor to the tiles, gather copies data from
   // the tiles to the tensor.
   enum TileDataOperation { Scatter, Gather };

   struct CopyTilesArgs {
       TiledTensor* tiledTensor;
       int start;
       int numTiles;
       TileDataOperation op;

       CopyTilesArgs(TiledTensor* _tiledTensor,
                     int _start,
                     int _numTiles,
                     TileDataOperation _op)
               : tiledTensor(_tiledTensor), start(_start), numTiles(_numTiles),
                 op(_op) {}
   };

   Tile* getTile(int index) { return &tiles[index]; }

   // Copy data (if needed) to this tile from the original tensor.
   void copyDataToTile(Tile* tile);

   // Copy data from this tile to the original tensor.
   void gatherDataFromTile(Tile* tile);

   // Split the work (data filling or gathering) and run them on multiple
   // threads.
   void parallelCopyTileData(TileDataOperation op);

   // True if we need to use the copyRawTensorData() for copying data.
   bool useRawTensor;

   // The original tensor that gets tiled into this TiledTensor.
   Tensor* origTensor;

   // True if all the tiles have data filled.
   bool dataFilled;

   std::vector<Tile> tiles;
};

}  // namespace smaug

#endif
