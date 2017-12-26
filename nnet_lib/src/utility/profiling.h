#ifndef _PROFILING_H_
#define _PROFILING_H_

// Performance profiling utility functions.
//
// There is naturally overhead associated with calling these functions, so the
// best way to profile something is to run it multiple times and amortize the
// measurement overhead. On gem5, I've found that the overhead of calling
// clock_gettime() ranges from 100-200 cycles.

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>

#include "core/nnet_fwd_defs.h"

typedef struct _string_t {
  char* str;
  unsigned len;
} string_t;

// A profiling entry structure.
//
// Pairs of calls to begin_profiling() and end_profiling() will create a new
// entry and push it onto this stack.
//
// Implemented as a stack as a singly linked-list.
struct _log_entry_t {
  string_t function_name;
  int layer_num;
  int invocation;
  uint64_t start_time;
  uint64_t end_time;
  struct _log_entry_t* next;
};
typedef struct _log_entry_t log_entry_t;

// Return the current timestamp counter via rdtscp.
//
// Remember to use this carefully! Results obtained from using this can be
// misleading if they don't take into account context switches, DVFS, and other
// system factors.
uint64_t get_cycle();

// Executes a serializing instruction.
void barrier();

// Return the current process's CPU time in nanoseconds via clock_gettime().
uint64_t get_nsecs();

// Start profiling.
//
// This creates a new profiling log entry and records the current cpu process
// time in this entry's start_time field along with the metadata in the call
// arguments.
void begin_profiling(const char* label, int layer_num);

// End profiling.
//
// This records the current time closes the last log entry's end_time field.
void end_profiling();

// Dumps all profiling logs to a file "profiling.log".
int dump_profiling_log();

// Writes profiling logs to the specified file pointer.
void write_profiling_log(FILE* out);

// Initialize the profiling system.
void init_profiling_log();

// Deletes all profiling logs.
void close_profiling_log();

#endif
