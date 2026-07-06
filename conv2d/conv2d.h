#pragma once

#include "layer.h"

class Conv2d : public Layer{
public:
    Conv2d(int in_channels, int in_height, int in_weight, int count_kernel, int size_kernel);
    std::vector<float> forward(const std::vector<float> &x) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;
    void update(float lr) override;
    void save_w(std::ofstream &in) override;
    void load_w(std::ifstream &out) override;

    double grad_squared_norm() const;   // сумма квадратов dK и dbias
    void scale_grad(float scale);       // умножить dK и dbias на scale

    int in_weight() const { return out_w; }
    int out_height() const { return out_h; }
private:
    int in_channels, in_w, in_h; //входные каналы, высота ширина
    int count_kernel, size_kernel; //кол-во ядер, их размер
    int out_w, out_h; //вых высота ширина


    std::vector<float> last_input;
    std::vector<std::vector<std::vector<std::vector<float>>>> K;
    std::vector<float> bias;

    std::vector<std::vector<std::vector<std::vector<float>>>> dK;
    std::vector<float> dbias;

    int in_idx(int ic, int u_, int v_) const { return (ic * in_h + u_) * in_w + v_ ;};
    int out_idx(int ic, int u_, int v_) const { return (ic * out_h + u_) * out_w + v_ ;};
};