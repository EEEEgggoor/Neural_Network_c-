#pragma once

#include "layer.h"


class MaxPool : public Layer{
public:
    MaxPool(int channels, int in_height, int in_weight, int pool_size);
    std::vector<float> forward(const std::vector<float> &x) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;

    int in_weight() const { return out_w; }
    int out_height() const { return out_h; }

private:
    int channels, in_w, in_h, pool_size;
    int out_h, out_w;

    std::vector<int> max_index;

    int in_idx(int ic, int u_, int v_) const { return (ic * in_h + u_) * in_w + v_ ;};
    int out_idx(int ic, int u_, int v_) const { return (ic * out_h + u_) * out_w + v_ ;};
};