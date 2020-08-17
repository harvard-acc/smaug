/**
 * \file globals.h
 * \brief SMAUG Global variables.
 */

#ifndef _CORE_GLOBALS_H_
#define _CORE_GLOBALS_H_

namespace smaug {

class ThreadPool;

/**
 * This is true if the user chooses to run the network in gem5 simulation.
 */
extern bool runningInSimulation;

/** True if we are simulating in fast-forward mode. */
extern bool fastForwardMode;

/**
 * The maximum number of accelerators an operator's work can be split across.
 * This limit exists to keep Aladdin simulation time and resources in check.
 */
constexpr const int maxNumAccelerators = 8;

/**
 * The actual number of accelerator complexes currently in use.
 */
extern int numAcceleratorsAvailable;

/**
 * The user-space thread pool used by SMAUG to run multithreaded tasks.
 */
extern ThreadPool* threadPool;

/**
 * If true, uses the systolic array for applicable operators when backend
 * support exists.
 */
extern bool useSystolicArrayWhenAvailable;

}  // namespace smaug

#endif
