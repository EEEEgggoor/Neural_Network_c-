#pragma once
#include "Layer.h"
#include <vector>


class FC_Layer : public Layer {
public:
    FC_Layer(int in_feat, int out_feat); //in_feat - кол-во на входе, in_feat - кол-во на выходе
    std::vector<float> forward(const std::vector<float> &x) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;
    std::vector<std::vector<float>> forward_T(const std::vector<float>& x);
    std::vector<float> backward_T(const std::vector<std::vector<float>>& grad_out);


    double grad_squared_norm() const;
    void scale_grad(float scale);

    template<class Visitor> void visit_parameters(Visitor visit) {
        for (int i = 0; i < out_feat; ++i) {
            for (int j = 0; j < in_feat; ++j) visit(W[i][j], dW[i][j]);
            visit(b[i], db[i]);
        }
    }

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

    std::vector<std::vector<float>> last_inputs_T;
};
