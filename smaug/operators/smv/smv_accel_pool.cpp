#include <string>

#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_accel_pool.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {

SmvAcceleratorPool::SmvAcceleratorPool(int _size)
        : size(_size), finishFlags(_size) {}

void SmvAcceleratorPool::addFinishFlag(
        int accelIdx, std::unique_ptr<volatile int> finishFlag) {
    if (runningInSimulation) {
        finishFlags[accelIdx].push_back(std::move(finishFlag));
    }
}

void SmvAcceleratorPool::join(int accelIdx) {
    if (finishFlags[accelIdx].empty())
        return;

    while (!finishFlags[accelIdx].empty()) {
        std::unique_ptr<volatile int> finishFlag =
                std::move(finishFlags[accelIdx].front());
        waitForAccelerator(finishFlag.get());
        finishFlags[accelIdx].pop_front();
    }
    dout(1) << "Accelerator " << accelIdx << " finished.\n";
}

void SmvAcceleratorPool::joinAll() {
    dout(1) << "Waiting for all accelerators to finish.\n";
    for (int i = 0; i < size; i++)
        join(i);
    dout(1) << "All accelerators finished.\n";
}

int SmvAcceleratorPool::getNextAvailableAccelerator(int currAccelIdx) {
    // Round-robin policy.
    int pickedAccel = currAccelIdx + 1;
    if (pickedAccel == size)
        pickedAccel = 0;
    // If the picked accelerator has not finished, wait until it returns.
    join(pickedAccel);
    if (size > 1)
        dout(1) << "Switched to accelerator " << pickedAccel << ".\n";
    return pickedAccel;
}

}  // namespace smaug
