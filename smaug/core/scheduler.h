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
    // After an operator is scheduled, this activates its children to be
    // scheduled if they become ready.
    void updateChildren(Operator* op);

    Network* network;
    Workspace* workspace;
    std::list<Operator*> readyQueue;
};

}  // namespace smaug
