#ifndef _CORE_GLOBALS_H_
#define _CORE_GLOBALS_H_

namespace smaug {
// This is true if the user chooses to run the network in gem5
// simulation.
extern bool runningInSimulation;
extern bool useSystolicArrayWhenAvailable;
}  // namespace smaug

#endif
