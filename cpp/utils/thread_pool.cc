// utils/thread_pool.cc
#include "utils/thread_pool.h"

namespace MERCURY {
    ThreadPool::ThreadPool(size_t min_threads, size_t max_threads) 
        : stop(false), min_threads(min_threads), max_threads(max_threads), active_threads(0) {
        
        // 기본 스레드 생성
        for(size_t i = 0; i < min_threads; ++i) {
            workers.emplace_back([this] { workerThread(false); });
        }
        
        last_expansion = std::chrono::steady_clock::now();
        last_cleanup = std::chrono::steady_clock::now();
    }

    ThreadPool::~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        
        // 기본 스레드 정리
        for(std::thread &worker: workers) {
            if(worker.joinable()) {
                worker.join();
            }
        }
        
        // 동적 스레드 정리
        for(std::thread &worker: dynamic_workers) {
            if(worker.joinable()) {
                worker.join();
            }
        }
    }
    
    bool ThreadPool::shouldExpandThreads() const {
        auto now = std::chrono::steady_clock::now();
        auto time_since_last_expansion = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_expansion);
        
        // 확장 조건:
        // 1. 대기 중인 작업이 현재 스레드 수보다 많음
        // 2. 최대 스레드 수에 도달하지 않음
        // 3. 최근 100ms 이내에 확장하지 않음 (너무 빈번한 확장 방지)
        size_t total_threads = workers.size() + dynamic_workers.size();
        bool queue_overloaded = tasks.size() > total_threads * 2;
        bool can_expand = total_threads < max_threads;
        bool expansion_cooldown = time_since_last_expansion.count() > 100;
        
        return queue_overloaded && can_expand && expansion_cooldown;
    }
    
    void ThreadPool::expandThreads() {
        // 한 번에 최대 2개씩 확장
        size_t threads_to_add = std::min(size_t(2), max_threads - workers.size() - dynamic_workers.size());
        
        for(size_t i = 0; i < threads_to_add; ++i) {
            dynamic_workers.emplace_back([this] { workerThread(true); });
        }
        
        last_expansion = std::chrono::steady_clock::now();
    }
    
    void ThreadPool::workerThread(bool is_dynamic) {
        auto thread_start = std::chrono::steady_clock::now();
        auto last_task_time = thread_start;
        const auto idle_timeout = std::chrono::seconds(30); // 30초 유휴 시간 후 동적 스레드 종료
        
        while(true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                
                // 동적 스레드의 경우 유휴 시간 체크
                if(is_dynamic) {
                    auto now = std::chrono::steady_clock::now();
                    auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(now - last_task_time);
                    
                    // 유휴 시간이 길거나 작업이 없으면 종료
                    if(idle_time > idle_timeout && tasks.empty()) {
                        // 동적 스레드 정리 목록에 추가
                        {
                            std::unique_lock<std::mutex> cleanup_lock(cleanup_mutex);
                            threads_to_cleanup.push_back(std::this_thread::get_id());
                        }
                        return; // 스레드 종료
                    }
                    
                    // 타임아웃과 함께 대기
                    if(!condition.wait_for(lock, std::chrono::milliseconds(1000), [this] { 
                        return stop || !tasks.empty(); 
                    })) {
                        continue; // 타임아웃 발생, 다시 유휴 시간 체크
                    }
                } else {
                    // 기본 스레드는 무한 대기
                    condition.wait(lock, [this] { 
                        return stop || !tasks.empty(); 
                    });
                }
                
                if(stop && tasks.empty()) return;
                
                if(!tasks.empty()) {
                    task = std::move(tasks.front());
                    tasks.pop();
                    last_task_time = std::chrono::steady_clock::now();
                    active_threads++;
                }
            }
            
            if(task) {
                task();
                active_threads--;
            }
        }
    }
    
    void ThreadPool::cleanupDynamicThreads() {
        std::unique_lock<std::mutex> cleanup_lock(cleanup_mutex);
        
        if(threads_to_cleanup.empty()) return;
        
        // 종료된 동적 스레드들을 정리
        auto it = dynamic_workers.begin();
        while(it != dynamic_workers.end()) {
            bool should_remove = false;
            
            for(auto cleanup_id : threads_to_cleanup) {
                if(it->get_id() == cleanup_id) {
                    if(it->joinable()) {
                        it->join();
                    }
                    should_remove = true;
                    break;
                }
            }
            
            if(should_remove) {
                it = dynamic_workers.erase(it);
            } else {
                ++it;
            }
        }
        
        threads_to_cleanup.clear();
        last_cleanup = std::chrono::steady_clock::now();
    }
}