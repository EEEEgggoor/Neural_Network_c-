#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>


inline std::vector<float> area_resize(const std::vector<float>& src, int srcW, int srcH,
                                       int dstW, int dstH) {
    std::vector<float> dst(dstW * dstH, 0.0f);
    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;

    for (int oy = 0; oy < dstH; oy++) {
        float sy0 = oy * scaleY;
        float sy1 = (oy + 1) * scaleY;
        int iy0 = std::max(0, static_cast<int>(std::floor(sy0)));
        int iy1 = std::min(srcH - 1, static_cast<int>(std::ceil(sy1)) - 1);
        if (iy1 < iy0) iy1 = iy0;

        for (int ox = 0; ox < dstW; ox++) {
            float sx0 = ox * scaleX;
            float sx1 = (ox + 1) * scaleX;
            int ix0 = std::max(0, static_cast<int>(std::floor(sx0)));
            int ix1 = std::min(srcW - 1, static_cast<int>(std::ceil(sx1)) - 1);
            if (ix1 < ix0) ix1 = ix0;

            float sum = 0.0f;
            int count = 0;
            for (int sy = iy0; sy <= iy1; sy++) {
                for (int sx = ix0; sx <= ix1; sx++) {
                    sum += src[sy * srcW + sx];
                    count++;
                }
            }
            dst[oy * dstW + ox] = count > 0 ? sum / count : 0.0f;
        }
    }
    return dst;
}

inline std::vector<float> load_image_as_mnist_input(const std::string& path, bool invert = true) {
    int width, height, channels;
    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 1);

    if (!img) {
        throw std::runtime_error("Не удалось загрузить изображение: " + path);
    }

    std::vector<float> src(width * height);
    for (int i = 0; i < width * height; i++) {
        float value = img[i] / 255.0f;
        if (invert) value = 1.0f - value;
        src[i] = value;
    }
    stbi_image_free(img);

    const float threshold = 0.15f;
    int minX = width, maxX = -1, minY = height, maxY = -1;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (src[y * width + x] > threshold) {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
        }
    }


    if (maxX < 0) {
        minX = 0; maxX = width - 1;
        minY = 0; maxY = height - 1;
    }

    int bboxW = maxX - minX + 1;
    int bboxH = maxY - minY + 1;

    std::vector<float> cropped(bboxW * bboxH);
    for (int y = 0; y < bboxH; y++)
        for (int x = 0; x < bboxW; x++)
            cropped[y * bboxW + x] = src[(minY + y) * width + (minX + x)];


    int longSide = std::max(bboxW, bboxH);
    float scale = 20.0f / longSide;
    int newW = std::max(1, static_cast<int>(std::round(bboxW * scale)));
    int newH = std::max(1, static_cast<int>(std::round(bboxH * scale)));

    std::vector<float> resized = area_resize(cropped, bboxW, bboxH, newW, newH);


    double sumMass = 0.0, sumX = 0.0, sumY = 0.0;
    for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {
            float v = resized[y * newW + x];
            sumMass += v;
            sumX += v * x;
            sumY += v * y;
        }
    }
    float comX = sumMass > 0 ? static_cast<float>(sumX / sumMass) : newW / 2.0f;
    float comY = sumMass > 0 ? static_cast<float>(sumY / sumMass) : newH / 2.0f;


    std::vector<float> result(28 * 28, 0.0f);
    int offX = static_cast<int>(std::round(13.5f - comX));
    int offY = static_cast<int>(std::round(13.5f - comY));

    for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {
            int dx = x + offX;
            int dy = y + offY;
            if (dx >= 0 && dx < 28 && dy >= 0 && dy < 28) {
                result[dy * 28 + dx] = resized[y * newW + x];
            }
        }
    }

    return result;
}