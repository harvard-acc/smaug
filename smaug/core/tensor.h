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

/**
 * TensorShape describes the shape of a Tensor.
 *
 * A Tensor's shape is described by three parameters: its dimensions, alignment
 * padding, and a DataLayout.
 *
 * 1. Dimensions: an N dimensional vector. The outermost dimension is at index
 *    0, and innermost dimemnsion is at index N  -1.
 * 2. Alignment: used to zeropad the innermost dimension to be a multiple of a
 *    hardware backend's data alignment requirements (for example, if an
 *    accelerator expects 8-wide vectors, the innermost dimension must be a
 *    multiple of 8). This can be zero if no alignment is required.
 * 3. DataLayout: the meaning of each dimension, e.g. "NCHW" means
 *    dimensions[0] = N, dimensions[3] = W, etc.
 */
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
    /** Returns a vector of padding along each dimension. */
    const std::vector<int>& padding() const { return padding_; }
    int operator[](int index) const { return dims_[getIndex(index)]; }
    int& operator[](int index) { return dims_[getIndex(index)]; }
    /** Returns the alignment-padded size of the specified dimension. */
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

    /** Return a TensorShapeProto that serializes this TensorShape. */
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
    /** Padding along each dimension. Only the last element be nonzero. */
    std::vector<int> padding_;
    DataLayout layout;
    int alignment;
};

/**
 * An iterator over a multidimensional tensor's indices, accounting for data
 * alignment padding.
 *
 * The iterator tracks the current location as a coordinate and outputs the
 * linearized index so that the data in a tensor can be accessed. While most
 * commonly used to iterate through the contents of a tensor one by one, it can
 * also provide random access to any location in the tensor.
 *
 * Example usage for simple iteration:
 *   auto iter = TensorIndexIterator(tensor->getShape());
 *    * OR: auto iter = tensor->startIndex();
 *   float* data = tensor->data<float>();
 *   while (!iter.end())
 *      std::cout << data[iter] << ",";
 *
 * Example usage for random access (assume 4D tensor):
 *   auto iter = TensorIndexIterator(tensor->getShape());
 *   float* data = tensor->data<float>();
 *   data[iter(1,2,3,4)] = 1.2;
 *   data[iter(3,4,0,0)] = 3.4;
 *
 * The iterator skips over data alignment padding areas, if any exist.
 */
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

    /** This returns the current index of the iterator on the specified dim. */
    int currentIndex(int dim) const { return state[dim]; }

   protected:
    /**
     * Returns the linear index into the Tensor's underlying data container at
     * the specified coordinates.
     */
    template <typename Container>
    int getIndex(Container indices) const {
        int linearIndex = 0, stride = 1;
        for (int i = (int)indices.size() - 1; i >= 0; i--) {
            linearIndex += indices[i] * stride;
            stride *= (dims.at(i) + padding.at(i));
        }
        return linearIndex;
    }

    /*
     * Advance the current iterator position by the given region size.
     *
     * @param region An N-dim vector indicating how far to increment in each
     * dimension, if the previous dimension overflowed and caused a carry-over
     * into the next dimension.
     */
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

    /** The current location of the iterator. */
    std::vector<int> state;
    /** The dimensions of this iterator's Tensor. */
    std::vector<int> dims;
    /** Alignment padding of the Tensor. */
    std::vector<int> padding;
    /** If true, we've reached the end of the Tensor. */
    bool atEnd;
    /** A vector of all ones, used to implement operator++. */
    const std::vector<int> advanceOne;
};

/**
 * A tensor index iterator that stays within a specified rectangular region.
 *
 * The rectangular region is specified using an origin coordinate and a region
 * size. The iterator will output linear indices in the same space as the full
 * tensor index iterator, but indices outside the region will be skipped.
 *
 * Example: consider a 3x3 tensor. The upper right 2x2 region's origin is at
 * location (0,1). We can output just that block like so:
 *
 *    auto it = TensorRegionIndexIterator(tensor->getShape(), {0,1}, {2,2});
 *    while (!it.end())
 *       std::cout << (int)it << "\n";
 *
 *  This produces: 1, 2, 4, 5
 */
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
    /** Advance the tensor region index with the specified region size. */
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

/**
 * The base class of all Tensor objects.
 *
 * This contains common properties used by all Tensor implementations, like
 * name, shape, and data type, but it does not contain the data itself.
 * Subclasses are responsible for designing, allocating, and managing data
 * storage.
 */
class TensorBase {
   public:
    TensorBase() : name(""), dataFormat(UnknownStorageFormat), dead(false) {}
    virtual ~TensorBase() {}

    TensorBase(const std::string& _name, const TensorShape& _shape)
            : name(_name), shape(_shape), dataFormat(Uncompressed),
              dataType(UnknownDataType), dead(false) {}

    TensorBase(const TensorProto& tensorProto)
            : name(tensorProto.name()), shape(tensorProto.shape()),
              dataFormat(tensorProto.data_format()),
              dataType(tensorProto.data_type()), dead(false) {}

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
                return 0;
        }
    }
    bool isDead() const { return dead; }
    void setDead(bool _dead = true) { dead = _dead; }
    virtual bool containsData() const = 0;

   protected:
    /** Name of of the Tensor. This should be a unique in the Workspace. */
    std::string name;
    /** Shape of the Tensor. */
    TensorShape shape;
    /**
     * Indicates the compression format of the data.
     * NOTE: Compressed data formats are not currently supported in SMAUG.
     */
    DataStorageFormat dataFormat;
    DataType dataType;
    /**
     * If true, the tensor is dead, which means it is on an untaken control
     * flow path. All operators that consume this tensor will eventually be
     * marked dead (except for MergeOp).
     */
    bool dead;
};

/**
 * Tensor represents a single multi-dimensional array of data.
 *
 * At construction time, the dataType of the Tensor is undetermined, and memory
 * to store its data is not allocated. The dataType is set by a call to
 * Tensor::allocateStorage() or Tensor::fillData<T>, where T is the type of the
 * incoming data. Afterwards, the underlying data array can be accessed with
 * Tensor::data<T> (which will check that T matches the expected data type and
 * assert-fail if not) and indexed using TensorIndexIterator.
 */
class Tensor : public TensorBase {
   public:
    Tensor() : TensorBase(), tensorData(NULL) {}

    /** Construct a Tensor with the given name and shape. */
    Tensor(const std::string& _name, const TensorShape& _shape)
            : TensorBase(_name, _shape), tensorData(NULL) {}
    virtual ~Tensor() {}

    /**
     * Constructs a Tensor from serialized protobufs.
     *
     * @param tensorProto Basic parameters of the Tensor.
     * @param tensorData The data contents of the Tensor.
     */
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
                break;
            case Bool:
                fillData<bool>(tensorData.bool_data());
                break;
            default:
                assert(false && "Unknown data format!");
        }
    }

    /** Returns an iterator starting at the beginning of the Tensor. */
    TensorIndexIterator startIndex() const {
        return TensorIndexIterator(shape);
    }

    virtual bool containsData() const { return tensorData != nullptr; }

    /**
     * Fills the Tensor with externalData.
     *
     * The contents of externalData are copied over byte by byte, without
     * regard for padding zones.
     */
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

    /**
     * Fills the Tensor byte-by-byte from the given initializer list.
     */
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

    /**
     * Fills the Tensor byte-by-byte from a protobuf repeated field.
     */
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

    /**
     * Fill the tensor with float16 data.
     *
     * This special overload is required because the data stored in TensorProto
     * packs two float16 into one int32.
     */
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

    /**
     * Allocates memory to store Tensor data.
     *
     * @tparam T The type of data to store.
     */
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

    /**
     * Allocates memory to store Tensor data.
     *
     * @param _dataType The type of data to store.
     */
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

    /** Serializes this Tensor to a TensorProto. */
    TensorProto* asTensorProto();

    /**
     * Returns a const pointer to the Tensor data.
     */
    template <typename T>
    const T* data() const {
        assert(ToDataType<T>::dataType == dataType);
        return reinterpret_cast<T*>(tensorData.get());
    }

    /**
     * Returns a non-const pointer to the Tensor data.
     */
    template <typename T>
    T* data() {
        assert(ToDataType<T>::dataType == dataType);
        return reinterpret_cast<T*>(tensorData.get());
    }

    /**
     * Prints the contents of the Tensor to the given ostream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

   protected:
    std::shared_ptr<void> tensorData;
};

/* A multidimensional container of Tensors..
 *
 * Each of the tensors in a TiledTensor represents one tile or rectangular
 * section of a large tensor.  TileTensor can be iterated over via a
 * TensorIndexIterator just like ordinary Tensors, except each iteration
 * produces a Tensor instead of scalar data.
 */
class TiledTensor : public TensorBase {
  public:
   TiledTensor(Tensor* _origTensor = nullptr, bool _useRawTensor = false)
           : TensorBase(), origTensor(_origTensor), useRawTensor(_useRawTensor),
             dataFilled(false) {}
   /**
    * Construct a TiledTensor.
    *
    * @param shape The shape of the TiledTensor - that is, the number of tiles
    * in each dimension. Alignment padding is ignored here.
    * @param _origTensor The Tensor that is being tiled.
    * @param _useRawTensor If true, data from the original Tensor is memcpy'ed
    * into the all the tiles, instead of being copied element-wise from/to a
    * specific region. Only useful for broadcasting data into the tiles.
    */
   TiledTensor(const TensorShape& shape,
               Tensor* _origTensor = nullptr,
               bool _useRawTensor = false)
           : TensorBase("", shape), origTensor(_origTensor),
             useRawTensor(_useRawTensor), dataFilled(false) {
       tiles.resize(shape.size());
   }

   virtual bool containsData() const { return !tiles.empty(); }

   TensorIndexIterator startIndex() const { return TensorIndexIterator(shape); }

   /** Returns a const pointer to the Tensor at the given linear index. */
   const Tensor* operator[](int index) const { return tiles.at(index).tensor; }
   /** Returns a mutable reference to the Tensor at the given linear index. */
   Tensor*& operator[](int index) { return tiles[index].tensor; }
   int size() const { return shape.size(); }

   /**
    * Returns true if this TiledTensor is tiled along the N and H logical
    * dimensions.
    */
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

   /**
    * Returns a Tensor at the specified tile position, with data copied from
    * the original tensor.
    */
   Tensor* getTileWithData(int index);

   /**
    * Set the specified tile to the provided Tensor, and optionally copy data
    * into it.
    */
   void setTile(int index,
                const std::vector<int>& origin,
                Tensor* tensor,
                bool copyData);

   /** Copies data (if needed) to all the tiles from the original Tensor. */
   void copyDataToAllTiles();

   /** 
    * Copies data from the TiledTensor into the original Tensor. We name it
    * "untile" because what it does reverses the tiling process.
    */
   void untile();

   static void* tileCopyWorker(void* _args);

  protected:
   /**
    * A tile is a rectangular portion of a larger Tensor.
    */
   struct Tile {
       /** The new smaller Tensor of this tile. */
       Tensor* tensor;
       /** The tile's coordinate origins in the original tensor. */
       std::vector<int> origin;
       /** True if the tile has its origin set. */
       bool hasOrigin;
       /** True if we have copied data to this tile. */
       bool hasData;

       /** 
        * Construct a new blank Tile.
        *
        * Set the properties of this Tile using TiledTensor::setTile
        */
       Tile() : tensor(nullptr), origin(), hasOrigin(false), hasData(false) {}
   };

   /**
    * Specifies what to do with the data in the original Tensor and tiles.
    */
   enum TileDataOperation { 
     /** Copies data from a contiguous Tensor to the tiles. */
     Scatter, 
     /** Copies data from the tiles to a contiguous Tensor. */
     Gather 
   };

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

   /** Copy data (if needed) to this tile from the original Tensor. */
   void copyDataToTile(Tile* tile);

   /** Copy data from this tile to the original Tensor. */
   void gatherDataFromTile(Tile* tile);

   /** Split the work (data filling or gathering) across multiple threads. */
   void parallelCopyTileData(TileDataOperation op);

   /** True if we should use copyRawTensorData() for copying data. */
   bool useRawTensor;

   /** The original Tensor that was tiled into this TiledTensor. */
   Tensor* origTensor;

   /** True if all the tiles have data filled. */
   bool dataFilled;

   /** The list of Tiles, indexed using a TensorIndexIterator. */
   std::vector<Tile> tiles;
};

}  // namespace smaug

#endif
