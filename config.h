#pragma once

struct conv_params {
    int in_channels;
    int in_height;
    int in_width;      // было in_weight — небольшая опечатка, лучше поправить (weight — вес, width — ширина)
    int count_kernel;
    int size_kernel;

    // производные величины считаются сразу при создании структуры
    int out_height() const { return in_height - size_kernel + 1; }
    int out_width()  const { return in_width  - size_kernel + 1; }
    int out_channels() const { return count_kernel; }
};

struct pool_params {
    int channels;
    int in_height;
    int in_width;
    int pool_size;

    int out_height() const { return in_height / pool_size; }
    int out_width()  const { return in_width  / pool_size; }
    int out_channels() const { return channels; }
};

struct fc_params {
    int in_feat;
    int out_feat;
};

class Config {
public:
    conv_params conv1, conv2;
    pool_params pool1, pool2;
    fc_params fc1, fc2;

    Config(int input_channels, int input_height, int input_width,
           int conv1_filters, int conv1_kernel, int pool1_size,
           int conv2_filters, int conv2_kernel, int pool2_size,
           int fc1_out)
    {
        // conv1 — параметры входа приходят СНАРУЖИ (это исходные данные, MNIST 1x28x28)
        conv1 = { input_channels, input_height, input_width, conv1_filters, conv1_kernel };

        // pool1 — параметры входа берутся из ВЫХОДА conv1, а не набираются руками
        pool1 = { conv1.out_channels(), conv1.out_height(), conv1.out_width(), pool1_size };

        // conv2 — параметры входа берутся из ВЫХОДА pool1
        conv2 = { pool1.out_channels(), pool1.out_height(), pool1.out_width(), conv2_filters, conv2_kernel };

        // pool2 — из выхода conv2
        pool2 = { conv2.out_channels(), conv2.out_height(), conv2.out_width(), pool2_size };

        // fc1 — in_feat считается из выхода pool2 (каналы * высота * ширина - вот тут точное 16*5*5=400 больше не пишем руками!)
        int flat_size = pool2.out_channels() * pool2.out_height() * pool2.out_width();
        fc1 = { flat_size, fc1_out };

        // fc2 — in_feat из выхода fc1, out_feat = 10 классов (фиксировано, не гиперпараметр)
        fc2 = { fc1.out_feat, 10 };
    }
};