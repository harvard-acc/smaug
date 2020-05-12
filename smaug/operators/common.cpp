#include "smaug/core/globals.h"
#include "smaug/operators/common.h"

namespace smaug {

std::string getTraceName(int accelIdx) {
    std::string traceName =
            "dynamic_trace_acc" + std::to_string(accelIdx) + ".gz";
    return traceName;
}

void mapArrayToAccel(unsigned reqCode,
                     const char* arrayName,
                     void* baseAddr,
                     size_t size) {
    if (runningInSimulation) {
        mapArrayToAccelerator(reqCode, arrayName, baseAddr, size);
    }
}

void setArrayMemTypeIfSimulating(unsigned reqCode,
                                 const char* arrayName,
                                 MemoryType memType) {
    if (runningInSimulation) {
        setArrayMemoryType(reqCode, arrayName, memType);
    }
}

}  // namespace smaug

#ifdef __cplusplus
extern "C" {
#endif

ALWAYS_INLINE
size_t next_multiple(size_t request, size_t align) {
    size_t n = request / align;
    if (n == 0)
        return align;  // Return at least this many bytes.
    size_t remainder = request % align;
    if (remainder)
        return (n + 1) * align;
    return request;
}

#ifdef __cplusplus
}
#endif
