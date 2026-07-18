#include "offline_asr/utils/thread_pool.h"

namespace offline_asr {

ThreadPool::ThreadPool(size_t num_threads) {
  if (num_threads == 0)
    num_threads = std::thread::hardware_concurrency();
  if (num_threads < 1)
    num_threads = 1;

  workers_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back(&ThreadPool::WorkerLoop, this);
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &w : workers_) {
    if (w.joinable())
      w.join();
  }
}

void ThreadPool::WorkerLoop() {
  while (true) {
    std::packaged_task<void()> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty())
        return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    active_tasks_.fetch_add(1, std::memory_order_relaxed);
    task();
    active_tasks_.fetch_sub(1, std::memory_order_relaxed);
  }
}

}  // namespace offline_asr
