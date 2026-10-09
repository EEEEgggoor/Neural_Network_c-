#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>

// One optimizer per network; moments are reset when starting a new training run.
class Adam {
    std::vector<double> first, second;
    size_t step = 0;
public:
    template<class Network> void update(Network& net, float lr) {
        size_t count = 0;
        net.visit_parameters([&](float&, float g) {
            if (!std::isfinite(g)) throw std::runtime_error("Non-finite parameter gradient");
            ++count;
        });
        if (first.empty()) { first.assign(count, 0); second.assign(count, 0); }
        if (first.size() != count) throw std::runtime_error("Optimizer parameter count changed");
        ++step;
        const double c1 = 1 - std::pow(.9, double(step));
        const double c2 = 1 - std::pow(.999, double(step));
        size_t i = 0;
        net.visit_parameters([&](float& w, float g) {
            first[i] = .9 * first[i] + .1 * g;
            second[i] = .999 * second[i] + .001 * double(g) * g;
            w -= float(lr * (first[i] / c1) / (std::sqrt(second[i] / c2) + 1e-8));
            ++i;
        });
    }
};
