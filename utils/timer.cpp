#include "timer.h"
#include <iostream>
#include <iomanip>

void Timer::start(const std::string& label) {
    starts_[label] = Clock::now();
}

double Timer::stop(const std::string& label) {
    auto it = starts_.find(label);
    if (it == starts_.end()) return 0.0;
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - it->second).count();
    totals_[label] += ms;
    starts_.erase(it);
    return ms;
}

double Timer::elapsed(const std::string& label) const {
    auto it = starts_.find(label);
    if (it == starts_.end()) return totals_.count(label) ? totals_.at(label) : 0.0;
    return std::chrono::duration<double, std::milli>(Clock::now() - it->second).count();
}

void Timer::report() const {
    std::cout << "\n--- Timing Report ---\n";
    for (auto& [label, ms] : totals_)
        std::cout << std::left << std::setw(30) << label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
    std::cout << "---------------------\n";
}

void Timer::reset(const std::string& label) {
    starts_.erase(label);
    totals_.erase(label);
}
