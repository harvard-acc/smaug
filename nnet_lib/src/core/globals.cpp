#include "core/globals.h"

namespace smaug {
bool runningInSimulation;
int numAcceleratorsAvailable;
ThreadPool* threadPool = nullptr;
bool useSystolicArrayWhenAvailable;
}  // namespace smaug
