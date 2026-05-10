#pragma once
#include <chrono>
#include <string>
#include <map>

class Timer {
public:
    void start(const std::string& label = "default");
    double stop(const std::string& label = "default");  // returns elapsed ms
    double elapsed(const std::string& label = "default") const;
    void report() const;
    void reset(const std::string& label = "default");

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    std::map<std::string, TimePoint> starts_;
    std::map<std::string, double>    totals_;
};
