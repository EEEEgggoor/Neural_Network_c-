#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <clocale>
#include <string>
#include "nn.h"
#include "mnist_loader.h"
#include "softmax_ce.h"
#include "softmax_ce_ctc.h"
#include "greedly_decode.h"
#include "cvl_loader.h"


std::string label_to_string(const std::vector<int>& label) {
    std::string s;
    for (int d : label) s += std::to_string(d);
    return s;
}


int main() {
    std::cout << "Start...\n" << std::flush;
    CNN net;

    std::ifstream weights_check("nums_cnn_weights.bin");
    if (weights_check.good()) {
        weights_check.close();
        std::cout << "Loaded saved weights, training skipped.\n";
        net.load("nums_cnn_weights.bin");

        auto img = load_digit_string_image("my_digit.png", CANVAS_H, CANVAS_W);
        auto logits = net.forward(img);

        std::string predicted = decode(logits);
        std::cout << "Predicted digit: " << predicted << "\n";

    } else {

        std::vector<CVLSample> train = load_cvl_dataset("train");
        std::vector<CVLSample> test  = load_cvl_dataset("cvl-strings-eval");

        const float lr = 0.00005f;
        const int epochs = 5;

        // Максимальная норма градиента после clip
        const float grad_clip_norm = 1.0f;


        std::cout << "Start epoch...\n";

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int correct = 0;
            int used_steps = 0; // сколько шагов реально пошло в backward/update

            for (size_t step = 0; step < train.size(); ++step) {
                size_t i = step;

                // Пропускаем примеры, для которых меток физически больше
                if (2 * (int)train[i].label.size() + 1 > 98) continue;

                auto logits = net.forward(train[i].image);

                // Защита от уже испорченных (NaN/inf) логитов
                bool has_bad_logits = false;
                for (const auto& row : logits) {
                    for (float v : row) {
                        if (!std::isfinite(v)) {
                            has_bad_logits = true;
                            break;
                        }
                    }
                    if (has_bad_logits) break;
                }

                if (has_bad_logits) {
                    std::cout << "Error: NaN/inf in logits on step " << step << " - SKIP\n";
                    continue;
                }

                std::vector<std::vector<float>> grad;
                auto p = softmax_T(logits);
                float loss = ctc_loss_and_grad(p, train[i].label, grad);


                if (!std::isfinite(loss)) {
                    std::cout << "Skip step " << step << ", loss is NaN/inf\n";
                    continue;
                }

                net.clip_gradients(grad, grad_clip_norm);
                total_loss += loss;
                used_steps++;

                std::string predicted = decode(logits);
                if (predicted == label_to_string(train[i].label)) correct++;

                net.backward(grad);
                net.update(lr);

                if (step % 100 == 0) {
                    std::cout << "Epoch " << epoch << " sample " << step
                              << " loss: " << loss << "\n";
                }
            }

            std::cout << "Epoch " << epoch
                      << " avg loss: " << (used_steps ? total_loss / used_steps : 0.0f)
                      << " train accuracy: " << (100.0f * correct / train.size()) << "%\n";
        }

        int test_correct = 0;
        for (size_t i = 0; i < test.size(); ++i) {
            auto logits = net.forward(test[i].image);
            std::string predicted = decode(logits);
            if (predicted == label_to_string(test[i].label)) test_correct++;
        }
        std::cout << "Test accuracy: " << (100.0f * test_correct / test.size()) << "%\n";

        net.save("nums_cnn_weights.bin");
        std::cout << "Weights saved to nums_cnn_weights.bin\n";
    }
    std::getchar();
    return 0;
}