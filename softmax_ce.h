#pragma once

#include <cmath>
#include <vector>
#include <algorithm>


std::vector<float> softmax(const std::vector<float> &z){
    float max_z = *std::max_element(z.begin(), z.end());
    std::vector<float> exp_z(z.size());
    float sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
        exp_z[i] = std::exp(z[i] - max_z);
        sum += exp_z[i];
    }
    for (auto& v : exp_z) v /= sum;
    return exp_z;
}

float cross_entropy_loss_and_grad(const std::vector<float>& z, int label, std::vector<float> &grad_out) {
    std::vector<float> p = softmax(z);
    float loss = -std::log(p[label] + 1e-9f);

    grad_out = p;
    grad_out[label] -= 1.0f;

    return loss;
}