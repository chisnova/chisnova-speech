// utils/thread_pool.h
#ifndef MECURY_THREAD_POOL_H
#define MECURY_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <atomic>
#include <chrono>
#include "logger/logger_manager.h"

namespace MERCURY {
    class ThreadPool {
    public:
        // 싱글톤 패턴: 전역에서 공유되는 스레드 풀 인스턴스
        static ThreadPool& getInstance() {
            static ThreadPool instance(4, 16); // 기본 4개, 최대 16개 스레드
            return instance;
        }
        
        // 복사 생성자와 할당 연산자 삭제 (싱글톤 보장)
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        
        // 소멸자: 모든 스레드 종료
        ~ThreadPool();

        // 작업 추가: 템플릿을 사용하여 다양한 반환 타입 지원
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args)
            -> std::future<typename std::result_of<F(Args...)>::type> {
            using return_type = typename std::result_of<F(Args...)>::type;
            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
            std::future<return_type> res = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");

                tasks.emplace([task](){ (*task)(); });

                // Cheap pre-check before expensive clock read:
                // only consider expansion if queue is substantially backed up
                if(tasks.size() > (workers.size() + dynamic_workers.size()) * 2) {
                    if(shouldExpandThreads()) {
                        expandThreads();
                    }
                }
            }
            condition.notify_one();
            return res;
        }

        // 스레드 풀 상태 확인
        bool is_stopped() const { return stop; }
        size_t get_thread_count() const { 
            std::unique_lock<std::mutex> lock(queue_mutex);
            return workers.size(); 
        }
        size_t get_queue_size() const { 
            std::unique_lock<std::mutex> lock(queue_mutex);
            return tasks.size(); 
        }
        size_t get_active_threads() const {
            return active_threads.load();
        }

        // 편의 함수: 간단한 접근을 위한 static 메서드
        template<class F, class... Args>
        static auto submit(F&& f, Args&&... args) 
            -> std::future<typename std::result_of<F(Args...)>::type> {
            return getInstance().enqueue(std::forward<F>(f), std::forward<Args>(args)...);
        }
        
        // 동적 스레드 정리 (외부에서 호출 가능)
        void cleanupDynamicThreads();

    private:
        // private 생성자 (싱글톤 패턴)
        ThreadPool(size_t min_threads, size_t max_threads);
        
        // 동적 스레드 관리
        bool shouldExpandThreads() const;
        void expandThreads();
        void shrinkThreads();
        void workerThread(bool is_dynamic = false);
        
        std::vector<std::thread> workers;
        std::vector<std::thread> dynamic_workers;
        std::queue<std::function<void()>> tasks;
        mutable std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic<bool> stop;
        
        // 동적 스레드 관리 변수
        size_t min_threads;
        size_t max_threads;
        std::atomic<size_t> active_threads;
        std::chrono::steady_clock::time_point last_expansion;
        std::chrono::steady_clock::time_point last_cleanup;
        
        // 동적 스레드 정리를 위한 변수
        std::vector<std::thread::id> threads_to_cleanup;
        std::mutex cleanup_mutex;
    };
}

#endif