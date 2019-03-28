#include "core/globals.h"
#include "operators/common.h"

namespace smaug {
void mapArrayToAccel(unsigned reqCode,
                     const char* arrayName,
                     void* baseAddr,
                     size_t size) {
    if (runningInSimulation) {
        mapArrayToAccelerator(reqCode, arrayName, baseAddr, size);
    }
}
}  // namespace smaug
