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
#include <stdexcept>


constexpr int CANVAS_H = 32;
constexpr int CANVAS_W = 400;

struct CNN {
    Conv2d conv1;     // 1x32x400  -> 8x30x398
    ReLu relu1;
    MaxPool pool1;    // 8x30x398  -> 8x15x199
    Conv2d conv2;     // 8x15x199  -> 16x13x197
    ReLu relu2;
    MaxPool pool2;    // 16x13x197 -> 16x6x98
    FC_Layer fc1;     // 16x6 -> 11 классов
    CNN()
        : conv1(1, CANVAS_H, CANVAS_W, 8, 3),
          pool1(8, 30, 398, 2),
          conv2(8, 15, 199, 16, 3),
          pool2(16, 13, 197, 2),
          fc1(16 * 6, 11)
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
            throw std::runtime_error("Non-finite CTC gradient");
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
            throw std::runtime_error("Non-finite parameter gradient");
        }

        if (norm > max_norm) {
            float scale = (float)(max_norm / norm);
            conv1.scale_grad(scale);
            conv2.scale_grad(scale);
            fc1.scale_grad(scale);
        }
    }

    std::vector<std::vector<float>> forward(const std::vector<float>& x) {
        if (x.size() != CANVAS_H * CANVAS_W)
            throw std::invalid_argument("CNN input must be 32 x 400");

        int last_ink = 0;
        for (int h = 0; h < CANVAS_H; ++h)
            for (int w = 0; w < CANVAS_W; ++w)
                if (x[h * CANVAS_W + w] > 0.05f) last_ink = std::max(last_ink, w);
        const int time_steps = std::min(98, (last_ink + 3) / 4 + 3);
        auto a = conv1.forward(x);
        a = relu1.forward(a);
        a = pool1.forward(a);
        a = conv2.forward(a);
        a = relu2.forward(a);
        a = pool2.forward(a);
       
        std::vector<float> sequence(time_steps * 96);
        for (int t = 0; t < time_steps; ++t)
            for (int c = 0; c < 16; ++c)
                for (int h = 0; h < 6; ++h)
                    sequence[t * 96 + c * 6 + h] = a[(c * 6 + h) * 98 + t];
        a = std::move(sequence);
        std::vector<std::vector<float>> r = fc1.forward_T(a);
        return r;
    }


    void backward(std::vector<std::vector<float>> _grad) {
        std::vector<float> grad = fc1.backward_T(_grad);
        grad.resize(98 * 96, 0.0f);
        std::vector<float> spatial(16 * 6 * 98);
        for (int t = 0; t < 98; ++t)
            for (int c = 0; c < 16; ++c)
                for (int h = 0; h < 6; ++h)
                    spatial[(c * 6 + h) * 98 + t] = grad[t * 96 + c * 6 + h];
        grad = std::move(spatial);
        grad = pool2.backward(grad);
        grad = relu2.backward(grad);
        grad = conv2.backward(grad);
        grad = pool1.backward(grad);
        grad = relu1.backward(grad);
        grad = conv1.backward(grad);
    }


    template<class Visitor> void visit_parameters(Visitor visit) {
        conv1.visit_parameters(visit);
        conv2.visit_parameters(visit);
        fc1.visit_parameters(visit);
    }

    void update(float lr) {
        conv1.update(lr);
        conv2.update(lr);
        fc1.update(lr);
    }

    void save(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write weights: " + path);
        out.write("NNCTC002", 8);
        conv1.save_w(out);
        conv2.save_w(out);
        fc1.save_w(out);
        if (!out) throw std::runtime_error("Failed writing weights: " + path);
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot read weights: " + path);
        char magic[8];
        if (!in.read(magic, 8) || std::string(magic, 8) != "NNCTC002")
            throw std::runtime_error("Old/incompatible weights. Train a new model with --train; old weights are preserved.");
        conv1.load_w(in);
        conv2.load_w(in);
        fc1.load_w(in);
        if (!in || in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Truncated or incompatible weights: " + path);
    }

};
