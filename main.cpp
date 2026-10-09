#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include "nn.h"
#include "adam.h"
#include "softmax_ce_ctc.h"
#include "greedly_decode.h"
#include "cvl_loader.h"

std::string label_to_string(const std::vector<int>& label) {
    std::string result;
    for (int digit : label) result += char('0' + digit);
    return result;
}

void print_prediction(const std::vector<std::vector<float>>& logits,
                      const std::vector<int>& label) {
    size_t blanks = 0;
    for (const auto& row : logits)
        blanks += std::max_element(row.begin(), row.end()) - row.begin() == 10;
    std::cout << " target=\"" << label_to_string(label)
              << "\" predicted=\"" << decode(logits)
              << "\" blank_steps=" << blanks << '/' << logits.size() << '\n';
}

size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<size_t> row(b.size() + 1);
    std::iota(row.begin(), row.end(), 0);
    for (size_t i = 1; i <= a.size(); ++i) {
        size_t diagonal = row[0]; row[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            size_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1,
                              diagonal + (a[i - 1] != b[j - 1])});
            diagonal = above;
        }
    }
    return row.back();
}

std::vector<CVLSample> load_samples(const std::string& folder, size_t limit) {
    if (!std::filesystem::is_directory(folder))
        throw std::runtime_error("Missing dataset directory: " + folder +
                                 ". Run from the project directory.");
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(folder))
        if (entry.is_regular_file() && entry.path().extension() == ".png") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    std::mt19937 random(42);
    std::shuffle(paths.begin(), paths.end(), random);
    std::vector<CVLSample> samples;
    for (const auto& path : paths) {
        auto label = label_from_filename(path.filename().string());
        if (label.empty()) continue;
        samples.push_back({load_digit_string_image(path.string(), CANVAS_H, CANVAS_W), std::move(label)});
        if (limit && samples.size() == limit) break;
    }
    if (samples.empty()) throw std::runtime_error("No labeled PNG images in " + folder);
    return samples;
}

void evaluate(CNN& net, const std::vector<CVLSample>& samples, const std::string& name) {
    size_t correct = 0, errors = 0, characters = 0, empty = 0;
    double loss = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        auto logits = net.forward(samples[i].image);
        auto expected = label_to_string(samples[i].label);
        auto predicted = decode(logits);
        correct += predicted == expected;
        empty += predicted.empty();
        errors += edit_distance(expected, predicted);
        characters += expected.size();
        std::vector<std::vector<float>> gradient;
        loss += ctc_logits_loss_and_grad(logits, samples[i].label, gradient);
        if (i < 10) { std::cout << name << " sample " << i; print_prediction(logits, samples[i].label); }
    }
    std::cout << name << " loss=" << loss / samples.size()
              << " exact_accuracy=" << 100.0 * correct / samples.size()
              << "% CER=" << 100.0 * errors / characters
              << "% empty=" << empty << '/' << samples.size() << std::endl;
}

int main(int argc, char** argv) {
    try {
        bool training = false, resume = false;
        int epochs = 20;
        size_t limit = 0, test_limit = 0;
        float lr = .001f;
        std::string weights = "nums_cnn_weights_v2.bin", output = "nums_cnn_weights_v2.bin", image;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i == argc) throw std::invalid_argument("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--train") training = true;
            else if (arg == "--resume") { training = true; resume = true; }
            else if (arg == "--epochs") epochs = std::stoi(value());
            else if (arg == "--lr") lr = std::stof(value());
            else if (arg == "--limit" || arg == "--test-limit") {
                int count = std::stoi(value());
                if (count < 0) throw std::invalid_argument("Limit must be nonnegative");
                (arg == "--limit" ? limit : test_limit) = size_t(count);
            }
            else if (arg == "--weights") weights = value();
            else if (arg == "--output") output = value();
            else if (arg == "--predict") image = value();
            else if (arg == "--help") {
                std::cout << "--train                 Train a fresh v2 model\n"
                          << "--resume                Load v2 weights and fine-tune (new Adam state)\n"
                          << "--epochs N --lr RATE    Defaults: 20, 0.001\n"
                          << "--limit N --test-limit N  Dataset limits; 0 means all\n"
                          << "--weights FILE --output FILE  Default: nums_cnn_weights_v2.bin\n"
                          << "--predict IMAGE        Recognize one image using saved weights\n"
                          << "No flags: evaluate saved v2 weights on train/test datasets.\n";
                return 0;
            } else throw std::invalid_argument("Unknown option: " + arg);
        }
        if (epochs <= 0 || !std::isfinite(lr) || lr <= 0)
            throw std::invalid_argument("Epochs and learning rate must be positive");
        if (training && !image.empty()) throw std::invalid_argument("Use --predict separately from training");
        CNN net;
        if (!training || resume) net.load(weights);
        if (!image.empty()) {
            auto logits = net.forward(load_digit_string_image(image, CANVAS_H, CANVAS_W));
            std::cout << "Predicted: \"" << decode(logits) << "\"\n";
            return 0;
        }
        auto train = load_samples("train", limit);
        auto test = load_samples("cvl-strings-eval", test_limit);
        if (training) {
            if (resume) std::cout << "Fine-tuning weights; Adam moments start from zero.\n";
            Adam optimizer;
            std::mt19937 random(42);
            std::vector<size_t> order(train.size());
            std::iota(order.begin(), order.end(), 0);
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(order.begin(), order.end(), random);
                double total_loss = 0;
                for (size_t step = 0; step < order.size(); ++step) {
                    const auto& sample = train[order[step]];
                    auto logits = net.forward(sample.image);
                    std::vector<std::vector<float>> gradient;
                    float loss = ctc_logits_loss_and_grad(logits, sample.label, gradient);
                    net.backward(gradient);
                    net.clip_param_gradients(5.0f);
                    optimizer.update(net, lr);
                    total_loss += loss;
                    if (step % 100 == 0) {
                        std::cout << "Epoch " << epoch + 1 << " sample " << step << " loss=" << loss;
                        print_prediction(logits, sample.label);
                    }
                }
                std::cout << "Epoch " << epoch + 1 << " mean training loss=" << total_loss / train.size() << std::endl;
            }
            net.save(output);
            std::cout << "Weights saved to " << output << '\n';
        }
        evaluate(net, train, "Train");
        evaluate(net, test, "Test");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
