#include <list>

#include "smaug/core/network.h"
#include "smaug/core/workspace.h"
#include "smaug/core/operator.h"

namespace smaug {

/**
 * Scheduler is responsible for running the Network.
 */
class Scheduler {
   public:
    Scheduler(Network* _network, Workspace* _workspace)
            : network(_network), workspace(_workspace) {}
    virtual ~Scheduler(){};
    /** Runs the Network to completion. The final output tensor is returned. */
    Tensor* runNetwork();

   protected:
    /**
     * Runs the operators in the ready queue. This may add new operators to
     * the ready queue by calling updateChildren().
     */
    Tensor* scheduleReady();

    /**
     * If none of the inputs to the current Operator are dead, then this will
     * run the Operator; otherwise, otherwise, all of the Operator's outputs
     * will be marked as dead tensors. The only exception is MergeOp, which can
     * run with dead inputs.
     */
    void maybeRunOperator(Operator* op);

    /**
     * After an Operator is run, this updates the number of pending inputs on
     * all its children. Any child Operator with no more pending inputs is then
     * added to the ready queue.
     */
    void updateChildren(Operator* op);

    Network* network;
    Workspace* workspace;

    /** The queue of all Operators ready to be executed. */
    std::list<Operator*> readyQueue;
};

}  // namespace smaug
