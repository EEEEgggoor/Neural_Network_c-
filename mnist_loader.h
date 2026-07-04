#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>


inline int32_t read_int32_be(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

struct MnistDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

inline MnistDataset load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ifstream img_file(images_path, std::ios::binary);
    std::ifstream lbl_file(labels_path, std::ios::binary);
    if (!img_file || !lbl_file)
        throw std::runtime_error("Не удалось открыть файлы MNIST");

    // Заголовок images-файла: magic number, число картинок, высота, ширина
    int32_t img_magic = read_int32_be(img_file);
    int32_t num_images = read_int32_be(img_file);
    int32_t rows = read_int32_be(img_file);
    int32_t cols = read_int32_be(img_file);

    // Заголовок labels-файла: magic number, число меток
    int32_t lbl_magic = read_int32_be(lbl_file);
    int32_t num_labels = read_int32_be(lbl_file);

    // Проверка сигнатуры - защита от повреждённого/неверного файла
    if (img_magic != 0x00000803 || lbl_magic != 0x00000801)
        throw std::runtime_error("Неверный формат файла MNIST");
    if (num_images != num_labels)
        throw std::runtime_error("Число картинок и меток не совпадает");

    MnistDataset dataset;
    dataset.images.resize(num_images);
    dataset.labels.resize(num_images);

    int image_size = rows * cols; // 28*28 = 784

    for (int i = 0; i < num_images; ++i) {
        // Пиксели идут подряд, по 1 байту (0-255) на пиксель, без разделителей
        std::vector<unsigned char> raw_pixels(image_size);
        img_file.read(reinterpret_cast<char*>(raw_pixels.data()), image_size);

        // Нормализация 0-255 -> 0.0-1.0: сеть обучается лучше на маленьких
        // числах, а Xavier-инициализация весов рассчитана на входы такого масштаба
        std::vector<float> pixels(image_size);
        for (int p = 0; p < image_size; ++p)
            pixels[p] = raw_pixels[p] / 255.0f;

        dataset.images[i] = std::move(pixels); // move вместо копии - экономим время на 60000 картинках

        unsigned char label_byte;
        lbl_file.read(reinterpret_cast<char*>(&label_byte), 1);
        dataset.labels[i] = static_cast<int>(label_byte);
    }

    return dataset;
}