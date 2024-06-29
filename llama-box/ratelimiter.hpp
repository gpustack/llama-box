#include <chrono>
#include <thread>

// lockless token bucket rate limiter
class token_bucket {

  private:
    int capacity;
    int rate;
    int tokens_remain;
    std::chrono::steady_clock::time_point last_time;

    void refill() {
        auto const now = std::chrono::steady_clock::now();
        auto const elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        int new_tokens = int(elapsed) * rate / 1000;
        if (new_tokens > 0) {
            tokens_remain = std::min(capacity, tokens_remain + new_tokens);
            last_time = now;
        }
    }

  public:
    token_bucket(int capacity, int rate)
        : capacity(capacity), rate(rate) {
        tokens_remain = capacity;
        last_time = std::chrono::steady_clock::now();
    }

    bool acquire(int tokens = 1) {
        if (this->tokens_remain < tokens) {
            refill();
            if (this->tokens_remain < tokens) {
                return false;
            }
        }
        this->tokens_remain -= tokens;
        return true;
    }
};