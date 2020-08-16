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

/**
 * Returns the elementwise-sum of the two arrays, which must be of the same
 * size.
 */
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

/**
 * Populates a std::vector with an arbitrary number of elements.
 */
template <typename T, typename... Args>
void variadicToVector(std::vector<T>& vector, T e, Args... elems) {
    vector.push_back(e);
    variadicToVector(vector, elems...);
}

/**
 * Returns a std::array populated with the given elements. Must contain at
 * least one element.
 *
 * @param i The first element.
 * @param elems All the remaining elements.
 */
template <typename T, typename... Args>
std::array<T, sizeof...(Args) + 1> variadicToArray(T i, Args... elems) {
    return {{ i, elems... }};
}

/**
 * Return heap-allocated cacheline-aligned memory.
 */
void* malloc_aligned(size_t size, bool zeroOut = false);

/**
 * Return the difference between value and the next multiple of alignment.
 */
int calc_padding(int value, unsigned alignment);

/** Get the string version of DataLayout. */
std::string dataLayoutToStr(DataLayout layout);

/**
 * Contains utility functions for interacting with gem5. In trace mode, these
 * are no-ops.
 */
namespace gem5 {

/**
 * Switches to the next CPU type. Often used to implement fast-forwarding, in
 * which switchCpu is called just before starting the detailed region of
 * simulation.
 */
void switchCpu();

/**
 * Dumps gem5 stats to the stats.txt file.
 *
 * @param msg A section label for this stats dump.
 * @param period Dump stats every N cycles. If 0, only dumps stats once.
 */
void dumpStats(const char* msg, int period = 0);

/** Dumps gem5 stats to the stats.txt file and resets all stats. */
void dumpResetStats(const char* msg, int period = 0);

/**
 * Puts the CPU to sleep. When a CPU is sleeping, it will not generate any
 * simulation events, respond to snoop packets, etc. It can be woken with a
 * call to wakeCpu.
 */
void quiesce();

/** Wakes up a quiesced CPU. */
void wakeCpu(int id);

/**
 * Returns the logical CPU number. This does not implement the cpuid
 * instruction!
 */
int getCpuId();

/**
 * A RAII helper class which dumps and/or resets gem5 stats at construction and
 * destruction.
 */
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
