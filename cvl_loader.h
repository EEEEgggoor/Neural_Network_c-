#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <filesystem>


struct CVLSample {
    std::vector<float> image;      // 32x400, из load_digit_string_image
    std::vector<int> label;        // например {1,3,5,5,7,9}
};


inline std::vector<float> area_resize(const std::vector<float>& src, int srcW, int srcH,
                                       int dstW, int dstH) {
    std::vector<float> dst(dstW * dstH, 0.0f);
    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;
    for (int oy = 0; oy < dstH; oy++) {
        float sy0 = oy * scaleY, sy1 = (oy + 1) * scaleY;
        int iy0 = std::max(0, (int)std::floor(sy0));
        int iy1 = std::min(srcH - 1, (int)std::ceil(sy1) - 1);
        if (iy1 < iy0) iy1 = iy0;
        for (int ox = 0; ox < dstW; ox++) {
            float sx0 = ox * scaleX, sx1 = (ox + 1) * scaleX;
            int ix0 = std::max(0, (int)std::floor(sx0));
            int ix1 = std::min(srcW - 1, (int)std::ceil(sx1) - 1);
            if (ix1 < ix0) ix1 = ix0;
            float sum = 0.0f; int count = 0;
            for (int sy = iy0; sy <= iy1; sy++)
                for (int sx = ix0; sx <= ix1; sx++) { sum += src[sy*srcW+sx]; count++; }
            dst[oy*dstW+ox] = count > 0 ? sum / count : 0.0f;
        }
    }
    return dst;
}


inline std::vector<float> load_digit_string_image(const std::string& path,
                                                   int canvas_h, int canvas_w,
                                                   bool invert = true) {
    int width, height, channels;
    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (!img) throw std::runtime_error("Не удалось загрузить изображение: " + path);

    std::vector<float> src(width * height);
    for (int i = 0; i < width * height; i++) {
        float v = img[i] / 255.0f;
        if (invert) v = 1.0f - v;
        src[i] = v;
    }
    stbi_image_free(img);


    const float threshold = 0.15f;
    int minX = width, maxX = -1, minY = height, maxY = -1;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (src[y*width+x] > threshold) {
                minX = std::min(minX, x); maxX = std::max(maxX, x);
                minY = std::min(minY, y); maxY = std::max(maxY, y);
            }
    if (maxX < 0) { minX = 0; maxX = width-1; minY = 0; maxY = height-1; }

    int bboxW = maxX - minX + 1;
    int bboxH = maxY - minY + 1;

    std::vector<float> cropped(bboxW * bboxH);
    for (int y = 0; y < bboxH; y++)
        for (int x = 0; x < bboxW; x++)
            cropped[y*bboxW+x] = src[(minY+y)*width + (minX+x)];


    float targetH = canvas_h * 0.8f;
    float scale = targetH / bboxH;
    int newH = std::max(1, (int)std::round(bboxH * scale));
    int newW = std::max(1, (int)std::round(bboxW * scale));


    if (newW > canvas_w) {
        float shrink = static_cast<float>(canvas_w) / newW;
        newW = canvas_w;
        newH = std::max(1, (int)std::round(newH * shrink));
    }

    std::vector<float> resized = area_resize(cropped, bboxW, bboxH, newW, newH);


    std::vector<float> canvas(canvas_h * canvas_w, 0.0f);
    int offY = (canvas_h - newH) / 2;
    int offX = 2;   // небольшой отступ слева, аналог полей в MNIST

    for (int y = 0; y < newH; y++)
        for (int x = 0; x < newW; x++) {
            int dy = y + offY, dx = x + offX;
            if (dy >= 0 && dy < canvas_h && dx >= 0 && dx < canvas_w)
                canvas[dy*canvas_w + dx] = resized[y*newW + x];
        }

    return canvas;
}


inline std::vector<int> label_from_filename(const std::string& filename) {
    std::vector<int> label;
    for (char c : filename) {
        if (c >= '0' && c <= '9') label.push_back(c - '0');
        else if (!label.empty()) break;   // цифры закончились - дальше служебная часть имени
    }
    return label;
}



std::vector<CVLSample> load_cvl_dataset(const std::string& folder) {
    std::vector<CVLSample> dataset;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (entry.path().extension() != ".png") continue;

        std::string filename = entry.path().filename().string();
        auto label = label_from_filename(filename);
        if (label.empty()) continue;   // на всякий случай пропустить файлы без цифр в имени

        auto image = load_digit_string_image(entry.path().string(), CANVAS_H, CANVAS_W);
        dataset.push_back({std::move(image), std::move(label)});
    }
    return dataset;
}