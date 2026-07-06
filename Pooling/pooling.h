#pragma once
#include "layer.h"

class Pooling : public Layer {
public:
    Pooling(int channels, int in_height, int in_weight);
    std::vector<float> forward_average_pooling(const std::vector<float> &x);
    std::vector<float> backward_average_pooling(const std::vector<float>& grad_out);

    std::vector<float> forward(const std::vector<float> &x) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;
public:
    int channels, in_w, in_h;
    int out_h, out_w;

    int in_idx(int ic, int u_, int v_) const { return (ic * in_h + u_) * in_w + v_ ;};
    int out_idx(int ic, int u_, int v_) const { return (ic * out_h + u_) * out_w + v_ ;};
    std::vector<float> transpose_C_W_to_W_C(const std::vector<float> _pooling);
    std::vector<float> transpose_W_C_to_C_W(const std::vector<float> _pooling);


};
