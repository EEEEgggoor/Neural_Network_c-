#pragma once
#include "conv2d/conv2d.h"
#include "ReLu/relu.h"
#include "MaxPool/max_pool.h"
#include "flatten.h"
#include "fc_layer.h"
#include "Pooling/pooling.h"
#include "softmax_ce.h"
#include <string>
#include <fstream>


constexpr int CANVAS_H = 32;
constexpr int CANVAS_W = 400;

struct CNN {
    Conv2d conv1;     // 1x32x400  -> 8x30x398
    ReLu relu1;
    MaxPool pool1;    // 8x30x398  -> 8x15x199
    Conv2d conv2;     // 8x15x199  -> 16x13x197
    ReLu relu2;
    MaxPool pool2;    // 16x13x197 -> 16x6x98
    Pooling pooling;  // схлопывает высоту 6 -> 1, остаётся 16 x 98 (T=98 шагов)
    FC_Layer fc1;     // 16 -> 11, применяется 98 раз (общие веса для каждого шага)
    CNN()
        : conv1(1, CANVAS_H, CANVAS_W, 8, 3),
          pool1(8, 30, 398, 2),
          conv2(8, 15, 199, 16, 3),
          pool2(16, 13, 197, 2),
          pooling(16, 6, 98),
          fc1(16, 11)
    {}

    void clip_gradients(std::vector<std::vector<float>>& grad, float max_norm) {
        double norm_sq = 0.0;
        bool has_nan = false;
        
        for (auto& row : grad)
            for (float v : row) {
                if (!std::isfinite(v)) { has_nan = true; break; }
                norm_sq += (double)v * v;
            }
        if (has_nan) {
            for (auto& row : grad)
                for (float& v : row) v = 0.0f;
            return;
        }
        double norm = std::sqrt(norm_sq);
        if (norm > max_norm) {
            float scale = (float)(max_norm / norm);
            for (auto& row : grad)
                for (float& v : row) v *= scale;
        }
    }

    void clip_param_gradients(float max_norm) {
        double norm_sq = conv1.grad_squared_norm()
                        + conv2.grad_squared_norm()
                        + fc1.grad_squared_norm();

        double norm = std::sqrt(norm_sq);

        if (!std::isfinite(norm)) {
            // градиенты весов уже испортились - обнуляем шаг целиком
            conv1.scale_grad(0.0f);
            conv2.scale_grad(0.0f);
            fc1.scale_grad(0.0f);
            return;
        }

        if (norm > max_norm) {
            float scale = (float)(max_norm / norm);
            conv1.scale_grad(scale);
            conv2.scale_grad(scale);
            fc1.scale_grad(scale);
        }
    }

    std::vector<std::vector<float>> forward(const std::vector<float>& x) {
        auto a = conv1.forward(x);
        a = relu1.forward(a);
        a = pool1.forward(a);
        a = conv2.forward(a);
        a = relu2.forward(a);
        a = pool2.forward(a);
        a = pooling.forward(a);       
        std::vector<std::vector<float>> r = fc1.forward_T(a);
        return r;
    }


    void backward(std::vector<std::vector<float>> _grad) {
        std::vector<float> grad = fc1.backward_T(_grad);
        grad = pooling.backward(grad);
        grad = pool2.backward(grad);
        grad = relu2.backward(grad);
        grad = conv2.backward(grad);
        grad = pool1.backward(grad);
        grad = relu1.backward(grad);
        grad = conv1.backward(grad);
    }


    void update(float lr) {
        conv1.update(lr);
        conv2.update(lr);
        fc1.update(lr);
    }

    void save(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        conv1.save_w(out);
        conv2.save_w(out);
        fc1.save_w(out);
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        conv1.load_w(in);
        conv2.load_w(in);
        fc1.load_w(in);
    }

};