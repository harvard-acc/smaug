#ifndef _UTILITY_THREAD_POOL_H_
#define _UTILITY_THREAD_POOL_H_

#include <pthread.h>
#include <vector>

namespace smaug {

class ThreadPool {
   public:
    ThreadPool(int nthreads);
    ~ThreadPool();

    typedef void* (*WorkerThreadFunc)(void*);

    int size() const { return workers.size(); }

    // Initialize the thread pool with nthreads. We can't do this right away in
    // the constructor because it is still in fast-forward mode and we will get
    // incorrect CPU IDs. Thus we need to call this after the CPUs are switched.
    // If this is called twice in a row, it will trigger an assertion failure.
    void initThreadPool();

    // Dispatch the thread to a worker in the thread pool.
    int dispatchThread(WorkerThreadFunc func, void* args);

    // Wait for all threads in the pool to return to idle state.
    void joinThreadPool();

   protected:
    enum ThreadStatus { Uninitialized, Idle, Running };
    struct WorkerThread {
        WorkerThreadFunc func;
        // These are provided by the user.
        void* args;

        // pthread handle.
        pthread_t thread;
        // This mutex protects all of the subsequent fields of this struct. Any
        // modification of these fields must first acquire this.
        pthread_mutex_t statusMutex;
        // Set to true to inform the worker thread to terminate.
        bool exit;
        // Set to true if the func and args are valid and need to be executed.
        bool valid;
        ThreadStatus status;
        // The main thread signals this condition variable to wake up the thread
        // and have it check for work (indicated by valid = true).
        pthread_cond_t wakeupCond;
        // The worker thread signals this condition variable to inform the main
        // thread of a change in status (usually from Running -> Idle).
        pthread_cond_t statusCond;
        // The gem5 simulation CPU ID corresponding to this worker thread.
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

    // This struct is only used to initialize the worker threads.
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

    static void* workerLoop(void* args);

    // Thread workers.
    std::vector<WorkerThread> workers;
};

}  // namespace smaug

#endif
