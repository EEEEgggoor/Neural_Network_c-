#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <clocale>


struct MnistDataset {
    std::vector<std::vector<float>> images; // каждая картинка - плоский вектор из 784 чисел [0,1]
    std::vector<int> labels;                // правильный класс (0-9) для каждой картинки
};

// Функция загружает датасет из CSV файла.
inline MnistDataset load_mnist_csv(const std::string& csv_path, bool has_header = true) {
    std::setlocale(LC_ALL, "Russian");

    std::ifstream file(csv_path);
    if (!file)
        throw std::runtime_error("Не удалось открыть CSV файл MNIST: " + csv_path);

    MnistDataset dataset;
    std::string line;
    size_t line_counter = 0;

    // Пропускаем строку заголовка, если она есть
    if (has_header) {
        if (std::getline(file, line)) {
            line_counter++;
        }
    }

    // Резервируем память (для train обычно 60000, для test - 10000)
    dataset.images.reserve(60000);
    dataset.labels.reserve(60000);

    while (std::getline(file, line)) {
        line_counter++;

        // Удаляем символ \r в конце строки, если файл в формате Windows (CRLF)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Пропускаем пустые строки
        if (line.empty()) continue;

        size_t start_pos = 0;
        size_t end_pos = line.find(',', start_pos);
        
        if (end_pos == std::string::npos) {
            // Если в строке вообще нет запятых — пропускаем или сигнализируем об ошибке
            continue; 
        }

        // 1. Читаем метку класса (первое число в строке)
        int label = 0;
        std::string label_token = line.substr(start_pos, end_pos - start_pos);
        try {
            label = std::stoi(label_token);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Ошибка stoi на строке " + std::to_string(line_counter) + 
                                     ". Не удалось распарсить метку класса: \"" + label_token + "\"");
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Ошибка stoi на строке " + std::to_string(line_counter) + 
                                     ". Значение метки выходит за границы int: \"" + label_token + "\"");
        }
        
        start_pos = end_pos + 1;

        // 2. Читаем 784 пикселя
        std::vector<float> pixels;
        pixels.reserve(784);

        while ((end_pos = line.find(',', start_pos)) != std::string::npos) {
            std::string pixel_token = line.substr(start_pos, end_pos - start_pos);
            try {
                float pixel_val = std::stof(pixel_token) / 255.0f;
                pixels.push_back(pixel_val);
            } catch (const std::exception& e) {
                throw std::runtime_error("Ошибка stof на строке " + std::to_string(line_counter) + 
                                         ". Не удалось распарсить пиксель: \"" + pixel_token + "\"");
            }
            start_pos = end_pos + 1;
        }

        // Не забываем про последний пиксель в строке (после него нет запятой)
        if (start_pos < line.size()) {
            std::string pixel_token = line.substr(start_pos);
            try {
                float pixel_val = std::stof(pixel_token) / 255.0f;
                pixels.push_back(pixel_val);
            } catch (const std::exception& e) {
                throw std::runtime_error("Ошибка stof на строке " + std::to_string(line_counter) + 
                                         ". Не удалось распарсить последний пиксель: \"" + pixel_token + "\"");
            }
        }

        // Защита от кривых данных
        if (pixels.size() != 784) {
            throw std::runtime_error("Ошибка на строке " + std::to_string(line_counter) + 
                                     ". Неверное количество пикселей. Ожидалось 784, получено " + std::to_string(pixels.size()));
        }

        // Сохраняем данные
        dataset.labels.push_back(label);
        dataset.images.push_back(std::move(pixels));
    }

    return dataset;
}