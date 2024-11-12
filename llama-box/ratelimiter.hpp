#include <chrono>
#include <thread>

// lockless token bucket rate limiter
class token_bucket {

  private:
    int rate;
    int tokens_remain;
    std::chrono::steady_clock::time_point last_time;

    void refill() {
        auto const now = std::chrono::steady_clock::now();
        auto elapsed   = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        if (elapsed < 1000) {
            elapsed = 0;
        }
        int new_tokens = (int(elapsed) / 1000) * rate;
        if (new_tokens > 0) {
            tokens_remain = std::min(capacity, tokens_remain + new_tokens);
            last_time     = now;
        }
    }

  public:
    int capacity;

    token_bucket(int capacity, int rate)
        : rate(rate), capacity(capacity) {
        tokens_remain = capacity;
        last_time     = std::chrono::steady_clock::now();
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