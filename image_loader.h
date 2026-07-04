#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vector>
#include <string>
#include <stdexcept>


inline std::vector<float> load_image_as_mnist_input(const std::string& path, bool invert = true) {
    int width, height, channels;
    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 1);

    if (!img) {
        throw std::runtime_error("Не удалось загрузить изображение: " + path);
    }

    std::vector<float> result(28 * 28, 0.0f);


    for (int oy = 0; oy < 28; oy++) {
        for (int ox = 0; ox < 28; ox++) {
            int src_x = ox * width / 28;
            int src_y = oy * height / 28;
            unsigned char pixel = img[src_y * width + src_x];

            float value = pixel / 255.0f;
            if (invert) value = 1.0f - value;  // см. пояснение ниже

            result[oy * 28 + ox] = value;
        }
    }

    stbi_image_free(img);
    return result;
}