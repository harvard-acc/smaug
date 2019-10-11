#ifndef _OPERATORS_SMV_SMV_ACCELERATOR_POOL_H_
#define _OPERATORS_SMV_SMV_ACCELERATOR_POOL_H_

#include <vector>
#include <deque>
#include <memory>

namespace smaug {

class SmvAcceleratorPool {
   public:
    SmvAcceleratorPool(int _size);

    // Add a finish flag for the specified accelerator.
    void addFinishFlag(int accelIdx, volatile int* finishFlag);

    // Wait until all the finish flags turn complete.
    void joinAll();

    // Get the next accelerator and wait if it's still busy. A round-robin
    // scheduling policy is used for picking the accelerator. The simple static
    // policy is used because the scheduling decisions need to be the same as
    // when we generate the traces, whereas dynamic decisions that depend on
    // runtime information may lead to mismatch between the traces and the
    // simulation.
    int getNextAvailableAccelerator(int currAccelIdx);

   protected:
    // Wait until this accelerator's finish flags turn complete.
    void join(int accelIdx);

    // Number of accelerators in the pool.
    int size;

    // Active finish flags for all the accelerators in the pool.
    std::vector<std::deque<std::unique_ptr<volatile int>>> finishFlags;
};

}  // namespace smaug

#endif
