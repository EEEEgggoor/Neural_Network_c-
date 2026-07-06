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
        const int epochs = 10;

        std::cout << "Start epoch...\n";
         
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int correct = 0;



            for (size_t step = 0; step < train.size(); ++step) {
                size_t i = step;

                auto logits = net.forward(train[i].image);

                std::vector<std::vector<float>> grad;
                auto p = softmax_T(logits);
                float loss = ctc_loss_and_grad(p, train[i].label, grad);
                net.clip_gradients(grad, 5.0f);
                total_loss += loss;

                std::string predicted = decode(logits);
                if (predicted == label_to_string(train[i].label)) correct++;

                net.backward(grad);
                net.update(lr);

                if (step % 5000 == 0)
                    std::cout << "Epoch " << epoch << " sample " << step
                            << " loss: " << loss << "\n";
            }

            std::cout << "Epoch " << epoch
                    << " avg loss: " << total_loss / train.size()
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