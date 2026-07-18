#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace offline_asr {

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  template <typename F, typename... Args>
  auto Enqueue(F &&f,
               Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>;

  size_t NumThreads() const { return workers_.size(); }

  int ActiveTasks() const {
    return active_tasks_.load(std::memory_order_relaxed);
  }

private:
  void WorkerLoop();

  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  std::atomic<int> active_tasks_{0};
};

template <typename F, typename... Args>
auto ThreadPool::Enqueue(F &&f, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  auto task = std::make_shared<std::packaged_task<ReturnType()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<ReturnType> result = task->get_future();

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (stop_)
      throw std::runtime_error("ThreadPool::Enqueue: pool has been stopped");
    tasks_.emplace([task]() { (*task)(); });
  }

  cv_.notify_one();
  return result;
}

}  // namespace offline_asr
