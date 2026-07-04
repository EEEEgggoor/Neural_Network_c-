#pragma once
#include "conv2d/conv2d.h"
#include "ReLu/relu.h"
#include "MaxPool/max_pool.h"
#include "flatten.h"
#include "fc_layer.h"
#include <string>
#include <fstream>


struct CNN {
    Conv2d conv1;      // 1x28x28  -> 8x26x26  
    ReLu relu1;
    MaxPool pool1;   // 8x26x26  -> 8x13x13  

    Conv2d conv2;      // 8x13x13  -> 16x11x11 
    ReLu relu2;
    MaxPool pool2;   // 16x11x11 -> 16x5x5

    FlattenLayer flatten; // 16x5x5 -> 400 

    FC_Layer fc1;       // 400 -> 64   
    ReLu relu3;
    FC_Layer fc2;       // 64 -> 10   

    CNN()
        : conv1(1, 28, 28, 8, 3),
          pool1(8, 26, 26, 2),
          conv2(8, 13, 13, 16, 3),
          pool2(16, 11, 11, 2),
          fc1(16 * 5 * 5, 64),
          fc2(64, 10)
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