#include "smaug/utility/thread_pool.h"
#include "smaug/utility/utils.h"
#include "smaug/core/globals.h"


namespace smaug {

ThreadPool::ThreadPool(int nthreads) : workers(nthreads) {}

ThreadPool::~ThreadPool() {
    // Shutdown the thread pool and free all resources.
    for (int i = 0; i < workers.size(); i++) {
        WorkerThread* worker = &workers[i];
        gem5::wakeCpu(worker->cpuid);
        pthread_mutex_lock(&worker->statusMutex);
        worker->exit = true;
        pthread_cond_signal(&worker->wakeupCond);
        pthread_mutex_unlock(&worker->statusMutex);
    }
    for (int i = 0; i < workers.size(); i++) {
        WorkerThread* worker = &workers[i];
        pthread_join(worker->thread, NULL);
        pthread_mutex_destroy(&worker->statusMutex);
        pthread_cond_destroy(&worker->wakeupCond);
        pthread_cond_destroy(&worker->statusCond);
    }
}

void* ThreadPool::workerLoop(void* args) {
    ThreadInitArgs* initArgs = reinterpret_cast<ThreadInitArgs*>(args);
    WorkerThread* worker = initArgs->worker;
    // Notify the main thread about this thread's cpuid. This can only be done
    // after the thread context is created.
    pthread_mutex_lock(&initArgs->cpuidMutex);
    initArgs->cpuid = gem5::getCpuId();
    worker->status = Idle;
    pthread_cond_signal(&initArgs->cpuidCond);
    pthread_mutex_unlock(&initArgs->cpuidMutex);

    do {
        gem5::quiesce();
        pthread_mutex_lock(&worker->statusMutex);
        while (!worker->valid && !worker->exit)
            pthread_cond_wait(&worker->wakeupCond, &worker->statusMutex);
        if (worker->valid) {
            worker->status = Running;
            pthread_mutex_unlock(&worker->statusMutex);

            // Run the function.
            worker->func(worker->args);

            pthread_mutex_lock(&worker->statusMutex);
            worker->status = Idle;
            worker->valid = false;
            pthread_cond_signal(&worker->statusCond);
        }
        bool exitThread = worker->exit;
        pthread_mutex_unlock(&worker->statusMutex);
        if (exitThread)
            break;
    } while (true);

    pthread_exit(NULL);
}

void ThreadPool::initThreadPool() {
    // Initialize the CPU ID for each worker thread.
    for (int i = 0; i < workers.size(); i++) {
        WorkerThread* worker = &workers[i];
        ThreadInitArgs initArgs(worker);
        pthread_create(
                &worker->thread, NULL, &ThreadPool::workerLoop, &initArgs);

        // Fill in the CPU ID of the worker thread.
        pthread_mutex_lock(&initArgs.cpuidMutex);
        while (initArgs.cpuid == -1 || worker->status == Uninitialized)
            pthread_cond_wait(&initArgs.cpuidCond, &initArgs.cpuidMutex);
        worker->cpuid = initArgs.cpuid;
        pthread_mutex_unlock(&initArgs.cpuidMutex);
        assert(worker->status != Uninitialized &&
               "Worker thread did not successfully initialize!");
    }
}

int ThreadPool::dispatchThread(WorkerThreadFunc func, void* args) {
    for (int i = 0; i < workers.size(); i++) {
        WorkerThread* worker = &workers[i];
        pthread_mutex_lock(&worker->statusMutex);
        if (worker->status == Idle && !worker->valid) {
            worker->func = func;
            worker->args = args;
            worker->valid = true;
            gem5::wakeCpu(worker->cpuid);
            pthread_cond_signal(&worker->wakeupCond);
            pthread_mutex_unlock(&worker->statusMutex);
            return i;
        }
        pthread_mutex_unlock(&worker->statusMutex);
    }
    return -1;
}

void ThreadPool::joinThreadPool() {
    // There is no need to call wakeCpu here. If the CPU is quiesced, then
    // it cannot possibly be running anything, so its status will be Idle, and
    // this will move on to the next CPU.
    for (int i = 0; i < workers.size(); i++) {
        WorkerThread* worker = &workers[i];
        pthread_mutex_lock(&worker->statusMutex);
        while (worker->status == Running || worker->valid == true)
            pthread_cond_wait(&worker->statusCond, &worker->statusMutex);
        pthread_mutex_unlock(&worker->statusMutex);
    }
}

}  // namespace smaug
