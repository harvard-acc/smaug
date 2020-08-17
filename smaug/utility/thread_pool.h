#ifndef _UTILITY_THREAD_POOL_H_
#define _UTILITY_THREAD_POOL_H_

#include <pthread.h>
#include <vector>

namespace smaug {

/**
 * A user-space cooperatve thread pool implementation designed for gem5 in SE
 * mode.
 *
 * Multithreading in gem5 SE mode is tricky - while we can spawn pthreads, we
 * cannot let threads terminate when the pthread function returns, because the
 * ThreadContext would be destroyed, which prevents us from ever using the CPU
 * that was assigned to that ThreadContext. The solution is to run an infinite
 * loop on all the threads in the pool and assign work to them from a queue.
 *
 * To prevent wasting simulation time with spinloops, this thread pool
 * implementation quiesces all inactive CPUs and wakes them up only when there
 * is work to do. This is done via magic gem5 instructions.
 */
class ThreadPool {
   public:
    /**
     * Create a ThreadPool with N threads.
     *
     * The simulation must be created with at least N+1 CPUs, since we need one
     * CPU to run the main thread.
     */
    ThreadPool(int nthreads);
    ~ThreadPool();

    /** Function signature for any work to be executed on a worker thread. */
    typedef void* (*WorkerThreadFunc)(void*);

    /** Returns the number of worker threads. */
    int size() const { return workers.size(); }

    /**
     * Initialize the thread pool.
     *
     * Initialization must be postponed until after fast-forwarding is
     * finished, or we will get incorrect CPU IDs.
     *
     * This can only be called once; any subsequent call will assert fail.
     */
    void initThreadPool();

    /** Dispatch the function to a worker in the thread pool. */
    int dispatchThread(WorkerThreadFunc func, void* args);

    /** Wait for all threads in the pool to finish work. */
    void joinThreadPool();

   protected:
    /** Possible worker thread states. */
    enum ThreadStatus { Uninitialized, Idle, Running };

    /** All state and metadata for a worker thread. */
    struct WorkerThread {
        WorkerThreadFunc func;
        /** User-provided arguments. */
        void* args;

        /** pthread handle. */
        pthread_t thread;
        /**
         * This mutex protects all of the subsequent fields of this struct. Any
         * modification of these fields must first acquire this lock.
         */
        pthread_mutex_t statusMutex;
        /** Set to true to inform the worker thread to terminate. */
        bool exit;
        /**  Set to true if the func and args are valid and need to be executed. */
        bool valid;
        ThreadStatus status;
        /**
         * The main thread signals this condition variable to wake up the
         * thread and have it check for work (indicated by valid = true).
         */
        pthread_cond_t wakeupCond;
        /** 
         * The worker thread signals this condition variable to inform the main
         * thread of a change in status (usually from Running -> Idle).
         */
        pthread_cond_t statusCond;
        /** The gem5 simulation CPU ID assigned to this worker thread. */
        int cpuid;

        WorkerThread() {
            func = NULL;
            args = NULL;
            exit = false;
            valid = false;
            status = Uninitialized;
            pthread_mutex_init(&statusMutex, NULL);
            pthread_cond_init(&wakeupCond, NULL);
            pthread_cond_init(&statusCond, NULL);
        }
    };

    struct ThreadInitArgs {
        WorkerThread* worker;
        pthread_mutex_t cpuidMutex;
        pthread_cond_t cpuidCond;
        int cpuid;

        ThreadInitArgs(WorkerThread* _worker) : worker(_worker) {
            pthread_mutex_init(&cpuidMutex, NULL);
            pthread_cond_init(&cpuidCond, NULL);
            cpuid = -1;
        }
    };

    /** The main event loop executed by all worker threads. */
    static void* workerLoop(void* args);

    /** Worker threads. */
    std::vector<WorkerThread> workers;
};

}  // namespace smaug

#endif
