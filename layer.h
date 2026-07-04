#pragma once
#include <vector>
#include <fstream>


class Layer {
public:
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual std::vector<float> backward(const std::vector<float>& grad_out) = 0;
    virtual void update(float lr) {};
    virtual void save_w(std::ofstream &in) {};
    virtual void load_w(std::ifstream &out) {};
    virtual ~Layer() = default;
};