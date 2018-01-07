#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>

#include "profiling.h"

static log_entry_t* profile_log;

// Indicates whether profiling has been enabled or not. If this is false, all
// calls to profiling functions are nops.
static bool profiling_enabled;

#ifdef __amd64

uint64_t get_cycle() {
    uint32_t hi, lo;
    __asm__("rdtscp" : "=a"(lo), "=d"(hi)::);
    return (uint64_t)(lo | ((uint64_t)(hi) << 32));
}

void barrier() {
    unsigned long long eax = 0, ebx = 0, ecx = 0, edx = 0;
    __asm__("cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(eax), "c"(ecx)
            :);
}

#else
#error "Cycle-level profiling on this architecture is not supported!"
#endif

uint64_t get_nsecs() {
  struct timespec time;
  uint64_t nsecs;
  barrier();
  int ret = clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time);
  if (ret) {
      perror("Unable to get process cpu time");
      return 0;
  }
  nsecs = time.tv_sec * 1e9 + time.tv_nsec;
  return nsecs;
}

void init_profiling_log() {
    profile_log = NULL;
    profiling_enabled = true;
}

// Allocate a new log entry.
log_entry_t* new_log_entry(const char* label, int layer_num, log_type type) {
    log_entry_t* entry = (log_entry_t*)malloc(sizeof(log_entry_t));
    entry->layer_num = layer_num;
    entry->invocation = 0;
    entry->type = type;
    entry->sample_data.sampled_iters = 1;
    entry->sample_data.total_iters = 1;

    // Copy the function name.
    entry->label.len = strlen(label) + 1;
    entry->label.str = (char*)malloc(entry->label.len * sizeof(char));
    strncpy(entry->label.str, label, entry->label.len - 1);
    entry->label.str[entry->label.len - 1] = 0;

    return entry;
}

// Push this new entry onto the stack.
void push_to_log_entry_stack(log_entry_t* entry, log_entry_t** stack_top) {
    if (stack_top == NULL) {
      entry->next = NULL;
      *stack_top = entry;
    } else {
      entry->next = *stack_top;
      *stack_top = entry;
    }
}

// Find the newest incomplete entry in the profiling log.
log_entry_t* find_newest_incomplete_entry() {
    log_entry_t* entry = profile_log;
    while (!(entry && entry->profile_data.end_time == 0))
      entry = entry->next;
    return entry;
}

void begin_profiling(const char* label, int layer_num) {
    if (!profiling_enabled)
        return;

    log_entry_t* entry = new_log_entry(label, layer_num, UNSAMPLED);
    push_to_log_entry_stack(entry, &profile_log);

    // Assign the rest of the metadata fields.
    entry->profile_data.end_time = 0;

    // Query the current time LAST, so it's as close as possible to the start
    // of the kernel being profiled.
    entry->profile_data.start_time = get_nsecs();
}

void end_profiling() {
    if (!profiling_enabled)
        return;
    log_entry_t* entry = find_newest_incomplete_entry();
		if (!entry || entry->profile_data.end_time != 0) {
        fprintf(stderr, "Could not find the corresponding entry for this "
                        "end_profiling call! Please ensure that all "
                        "begin_profiling() calls are paired with at most one "
                        "end_profiling call.\n");
        exit(1);
    }
    entry->profile_data.end_time = get_nsecs();
}

// Format is:
//
// num,label,function,invocation,start_time,end_time,elapsed_time
void write_profiling_log(FILE* out) {
    fprintf(out, "layer_num,label,invocation,type,start_time,end_time,elapsed_"
                 "time,sampled_iters,total_iters\n");
    log_entry_t* curr_entry = profile_log;
    while (curr_entry) {
        fprintf(out,
                "%d,%s,%d,%s,%lu,%lu,%lu,%ld,%ld\n",
                curr_entry->layer_num,
                curr_entry->label.str,
                curr_entry->invocation,
                curr_entry->type == UNSAMPLED ? "unsampled" : "sampled",
                curr_entry->profile_data.start_time,
                curr_entry->profile_data.end_time,
                curr_entry->profile_data.end_time -
                        curr_entry->profile_data.start_time,
                curr_entry->sample_data.sampled_iters,
                curr_entry->sample_data.total_iters);
        curr_entry = curr_entry->next;
    }
}

int dump_profiling_log() {
    if (!profiling_enabled)
        return 0;

    FILE* profile = fopen("profiling.log", "w");
    if (!profile) {
      perror("Unable to open profiling.log file");
      return -1;
    }
    write_profiling_log(profile);
    fclose(profile);
    return 0;
}

void close_profiling_log() {
    if (!profiling_enabled)
        return;

    log_entry_t* curr_entry = profile_log;
    while (curr_entry) {
        log_entry_t* next = curr_entry->next;
        free(curr_entry);
        curr_entry = next;
    }
    profile_log = NULL;
    profiling_enabled = false;
}

void set_profiling_type_sampled(int sampled_iters, int total_iters) {
    log_entry_t* entry = find_newest_incomplete_entry();
    entry->type = SAMPLED;
    entry->sample_data.sampled_iters = sampled_iters;
    entry->sample_data.total_iters = total_iters;
}
