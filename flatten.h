#pragma once
#include "layer.h"


class FlattenLayer : public Layer{
public:
    std::vector<float> forward(const std::vector<float> &x) { return x; };
    std::vector<float> backward(const std::vector<float>& grad_out) { return grad_out; };
};