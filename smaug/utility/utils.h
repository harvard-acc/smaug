#ifndef _UTILITY_UTILS_H_
#define _UTILITY_UTILS_H_

#include <array>
#include <string>
#include <vector>

#include "smaug/core/datatypes.h"
#include "gem5/m5ops.h"

namespace smaug {

// TODO: Allow these to take rvalue references.
template <typename T>
int product(std::vector<T> array) {
    int prod = 1;
    for (auto val : array)
        prod *= val;
    return prod;
}

template <typename T>
std::vector<T> sum(std::vector<T> array0, std::vector<T> array1) {
    assert(array0.size() == array1.size());
    std::vector<T> sum(array0.size());
    for (int i = 0; i < array0.size(); i++)
      sum[i] = array0[i] + array1[i];
    return sum;
}

template <typename T>
void variadicToVector(std::vector<T>& vector, T elem) {
    vector.push_back(elem);
}

template <typename T, typename... Args>
void variadicToVector(std::vector<T>& vector, T e, Args... elems) {
    vector.push_back(e);
    variadicToVector(vector, elems...);
}

template <typename T, typename... Args>
std::array<T, sizeof...(Args) + 1> variadicToArray(T i, Args... elems) {
    return {{ i, elems... }};
}

void* malloc_aligned(size_t size, bool zeroOut = false);

// Return the difference between @value and the next multiple of @alignment.
int calc_padding(int value, unsigned alignment);

std::string dataLayoutToStr(DataLayout layout);

namespace gem5 {

void switchCpu();
void dumpStats(const char* msg, int period = 0);
void dumpResetStats(const char* msg, int period = 0);
void quiesce();
void wakeCpu(int id);
int getCpuId();

class ScopedStats {
   public:
    ScopedStats(const char* _startLabel,
                const char* _endLabel,
                bool _resetStats = true);
    ~ScopedStats();

   protected:
    const char* startLabel;
    const char* endLabel;
    bool resetStats;
};

}  // namespace gem5

namespace stats {
constexpr const char* kNetworkStart = "Network start";
constexpr const char* kNetworkEnd = "Network end";
constexpr const char* kReorderingStart = "Reordering start";
constexpr const char* kReorderingEnd = "Reordering end";
constexpr const char* kTensorPrepStart = "Tensor preparation start";
constexpr const char* kTensorPrepEnd = "Tensor preparation end";
constexpr const char* kTensorFinalStart = "Tensor finalization start";
constexpr const char* kTensorFinalEnd = "Tensor finalization end";
}  // namespace stats

}  // namespace smaug

#endif
