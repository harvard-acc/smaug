#include "smaug/core/globals.h"

namespace smaug {
bool runningInSimulation;
bool fastForwardMode = true;
int numAcceleratorsAvailable;
ThreadPool* threadPool = nullptr;
bool useSystolicArrayWhenAvailable;
}  // namespace smaug
