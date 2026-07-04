#pragma once
#include "Layer.h"
#include <vector>


class FC_Layer : public Layer {
public:
    FC_Layer(int in_feat, int out_feat);
    std::vector<float> forward(const std::vector<float> &x) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;
    void update(float lr) override;
    void save_w(std::ofstream &in) override;
    void load_w(std::ifstream &out) override;

private:
    int in_feat, out_feat;
    std::vector<std::vector<float>> W;
    std::vector<float> b;

    std::vector<float> last_input;
    std::vector<std::vector<float>> dW;
    std::vector<float> db;
};