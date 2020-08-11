#ifndef _OPERATORS_SMV_SMV_ACCELERATOR_POOL_H_
#define _OPERATORS_SMV_SMV_ACCELERATOR_POOL_H_

#include <vector>
#include <deque>
#include <memory>

namespace smaug {

/**
 * Implements a pool of worker accelerators.
 *
 * For operators that require work tiling, tiles can be distributed across
 * multiple accelerators to exploit parallelism. This class implements a
 * deterministic round-robin worker pool. Determinism is required because when
 * generating multiple dynamic traces, worker accelerator assignments must
 * match with simulation of the binary in gem5.
 *
 * To use:
 *
 * ```c
 * SmvAcceleratorPool pool(size);
 * int currAccel = 0;
 * for (int i = 0; i < tiles; i++) {
 *    volatile int* finishFlag = invokeKernelNoBlock(currAccel, redCode, kernel, args...);
 *    pool.addFinishFlag(currAccel, std::make_unique(finishFlag));
 *    currAccel = pool.getNextAvailableAccelerator(currAccel);
 * }
 * pool.joinAll();
 * ```
 */
class SmvAcceleratorPool {
   public:
    SmvAcceleratorPool(int _size);

    /** Add a finish flag for the specified accelerator. */
    void addFinishFlag(int accelIdx, std::unique_ptr<volatile int> finishFlag);

    /** Wait until all the finish flags turn complete. */
    void joinAll();

    /**
     * Get the next accelerator and wait if it's still busy. A round-robin
     * scheduling policy is used for picking the accelerator. The simple static
     * policy is used because the scheduling decisions need to be the same as
     * when we generate the traces, whereas dynamic decisions that depend on
     * runtime information may lead to mismatch between the traces and the
     * simulation.
     *
     * TODO(xyzsam): the pool should be able to keep track of the current
     * accelerator index on its own.
     */
    int getNextAvailableAccelerator(int currAccelIdx);

   protected:
    /** Wait until this accelerator's finish flags turn complete. */
    void join(int accelIdx);

    /** Number of accelerators in the pool. */
    int size;

    /** Active finish flags for all the accelerators in the pool. */
    std::vector<std::deque<std::unique_ptr<volatile int>>> finishFlags;
};

}  // namespace smaug

#endif
