#pragma once
#include "layer.h"

class ReLu : Layer{
public:
    std::vector<float> forward(const std::vector<float> &x) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;

private:
    std::vector<float> last_input;
};