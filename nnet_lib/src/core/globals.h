#ifndef _CORE_GLOBALS_H_
#define _CORE_GLOBALS_H_

namespace smaug {

class ThreadPool;

// This is true if the user chooses to run the network in gem5
// simulation.
extern bool runningInSimulation;
// True if we are in the fast-forward mode.
extern bool fastForwardMode;
constexpr const int maxNumAccelerators = 8;
// The number of accelerators we have in total.
extern int numAcceleratorsAvailable;
extern ThreadPool* threadPool;
extern bool useSystolicArrayWhenAvailable;

}  // namespace smaug

#endif
