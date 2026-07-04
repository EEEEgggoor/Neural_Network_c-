#pragma once

#include "conv2d/conv2d.h"
#include "ReLu/relu.h"
#include "MaxPool/max_pool.h"
#include "flatten.h"
#include "fc_layer.h"
#include "config.h"
#include <string>
#include <fstream>


struct CNN {
    Conv2d conv1;
    ReLu relu1;
    MaxPool pool1;
    Conv2d conv2;
    ReLu relu2;
    MaxPool pool2;

    FlattenLayer flatten;

    FC_Layer fc1;
    ReLu relu3;
    FC_Layer fc2;
    
    Config cfg; 

    CNN() :
        cfg(1, 28, 28,      // вход: 1 канал, 28x28 (MNIST)
            8, 3, 2,        // conv1: 8 фильтров, ядро 3, pool 2
            16, 3, 2,       // conv2: 16 фильтров, ядро 3, pool 2
            64),            // fc1 выход 64

        conv1(cfg.conv1.in_channels, cfg.conv1.in_height, cfg.conv1.in_width,
              cfg.conv1.count_kernel, cfg.conv1.size_kernel),

        pool1(cfg.pool1.channels, cfg.pool1.in_height, cfg.pool1.in_width, cfg.pool1.pool_size),

        conv2(cfg.conv2.in_channels, cfg.conv2.in_height, cfg.conv2.in_width,
              cfg.conv2.count_kernel, cfg.conv2.size_kernel),

        pool2(cfg.pool2.channels, cfg.pool2.in_height, cfg.pool2.in_width, cfg.pool2.pool_size),

        fc1(cfg.fc1.in_feat, cfg.fc1.out_feat),
        fc2(cfg.fc2.in_feat, cfg.fc2.out_feat)
    {}


    std::vector<float> forward(const std::vector<float>& x) {
        auto a = conv1.forward(x);
        a = relu1.forward(a);
        a = pool1.forward(a);
        a = conv2.forward(a);
        a = relu2.forward(a);
        a = pool2.forward(a);
        a = flatten.forward(a);
        a = fc1.forward(a);
        a = relu3.forward(a);
        a = fc2.forward(a);
        return a;
    }


    void backward(std::vector<float> grad) {
        grad = fc2.backward(grad);
        grad = relu3.backward(grad);
        grad = fc1.backward(grad);
        grad = flatten.backward(grad);
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
        fc2.update(lr);
    }

    void save(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        conv1.save_w(out);
        conv2.save_w(out);
        fc1.save_w(out);
        fc2.save_w(out);
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        conv1.load_w(in);
        conv2.load_w(in);
        fc1.load_w(in);
        fc2.load_w(in);
    }
};