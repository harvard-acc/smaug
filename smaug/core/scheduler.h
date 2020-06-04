#include <list>

#include "smaug/core/network.h"
#include "smaug/core/workspace.h"
#include "smaug/core/operator.h"

namespace smaug {

class Scheduler {
   public:
    Scheduler(Network* _network, Workspace* _workspace)
            : network(_network), workspace(_workspace) {}
    virtual ~Scheduler(){};
    // Schedules every operator in the network.
    Tensor* runNetwork();

   protected:
    // Schedules the operators in the ready queue. This may add new operators to
    // the readu queue by calling updateChildren().
    Tensor* scheduleReady();

    // This will call the run() method of the given operator if none of its
    // inputs are dead, otherwise all its outputs will be marked as dead
    // tensors. An exception is the merge operator, which is the currently the
    // only operator that will run with dead inputs.
    void maybeRunOperator(Operator* op);

    // After an operator is scheduled, this activates its children to be
    // scheduled if they become ready.
    void updateChildren(Operator* op);

    Network* network;
    Workspace* workspace;
    std::list<Operator*> readyQueue;
};

}  // namespace smaug
