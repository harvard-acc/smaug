#ifndef _CORE_DATATYPES_H_
#define _CORE_DATATYPES_H_

#include <cstdint>
#include <vector>
#include "smaug/core/types.pb.h"

namespace smaug {

class DataLayoutSet {
  public:
    // DataLayoutSet(DataLayout layout) : layouts(layout) {}
    DataLayoutSet(int bitmask = 0) : layouts(bitmask) {}

    void insert(DataLayout layout) { layouts |= layout; }
    void remove(DataLayout layout) { layouts &= (~layout); }
    bool contains(DataLayout layout) const { return layouts & (int)layout; }
    bool overlapsWith(const DataLayoutSet& other) const {
        return (layouts & other.layouts) != 0;
    }
    int getLayouts() const { return layouts; }
    bool operator==(const DataLayoutSet& other) const {
        return layouts == other.layouts;
    }
    bool operator<(const DataLayoutSet& other) const {
        return layouts < other.layouts;
    }
    std::vector<DataLayout> toArray() const {
        int mask = DataLayout::EndDataLayout;
        std::vector<DataLayout> array;
        while (mask > 0) {
            int match = layouts & mask;
            if (match)
                array.push_back((DataLayout)mask);
            mask >>= 1;
        }
        return array;
    }

   protected:
    int layouts;
};

using float16 = uint16_t;

/**
 * ToDataType provides a compile-time way to convert a C type (e.g. double,
 * bool) to a SMAUG DataType enum (defined in smaug/core/types.proto).
 */
template <typename T>
struct ToDataType {};

template <>
struct ToDataType<float16> {
    static const DataType dataType = Float16;
};
template <>
struct ToDataType<float> {
    static const DataType dataType = Float32;
};
template <>
struct ToDataType<double> {
    static const DataType dataType = Float64;
};
template <>
struct ToDataType<int32_t> {
    static const DataType dataType = Int32;
};
template <>
struct ToDataType<uint32_t> {
    static const DataType dataType = Int32;
};
template <>
struct ToDataType<int64_t> {
    static const DataType dataType = Int64;
};
template <>
struct ToDataType<uint64_t> {
    static const DataType dataType = Int64;
};
template <>
struct ToDataType<bool> {
    static const DataType dataType = Bool;
};

template <enum DataType>
struct FromDataType {};

template<>
struct FromDataType<Int32> {
    typedef int32_t type;
};
template<>
struct FromDataType<Int64> {
    typedef int64_t type;
};
template<>
struct FromDataType<Float16> {
    typedef uint16_t type;
};
template<>
struct FromDataType<Float32> {
    typedef float type;
};
template<>
struct FromDataType<Float64> {
    typedef double type;
};
template<>
struct FromDataType<Bool> {
    typedef bool type;
};

}  // namespace smaug

#endif
