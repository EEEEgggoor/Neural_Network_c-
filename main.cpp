#include <iostream>
#include <algorithm>
#include <numeric>  
#include <random>   
#include <clocale>
#include "nn.h"
#include "mnist_loader.h"
#include "softmax_ce.h"
#include "image_loader.h"

int main() {
    std::cout << "Start...\n" << std::flush;

    MnistDataset train = load_mnist_csv("mnist_train.csv");

    MnistDataset test  = load_mnist_csv("mnist_test.csv");

    CNN net;

    std::ifstream weights_check("mnist_cnn_weights.bin");
    if (weights_check.good()) {
        weights_check.close();
        std::cout << "Loaded saved weights, training skipped.\n";
        net.load("mnist_cnn_weights.bin");

        auto img = load_image_as_mnist_input("my_digit.png");
        auto logits = net.forward(img);

        int predicted = std::max_element(logits.begin(), logits.end()) - logits.begin();
        std::cout << "Predicted digit: " << predicted << "\n";
        

    } else {
        const float lr = 0.01f;
        const int epochs = 3;

        std::vector<size_t> indices(train.images.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::cout << "Start epoch...\n";
         
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int correct = 0;

            std::shuffle(indices.begin(), indices.end(), g);

            for (size_t step = 0; step < train.images.size(); ++step) {
                size_t i = indices[step];

                auto logits = net.forward(train.images[i]);

                std::vector<float> grad;
                float loss = cross_entropy_loss_and_grad(logits, train.labels[i], grad);
                total_loss += loss;

                int predicted = std::max_element(logits.begin(), logits.end()) - logits.begin();
                if (predicted == train.labels[i]) correct++;

                net.backward(grad);
                net.update(lr);

                if (step % 5000 == 0)
                    std::cout << "Epoch " << epoch << " sample " << step
                            << " loss: " << loss << "\n";
            }

            std::cout << "Epoch " << epoch
                    << " avg loss: " << total_loss / train.images.size()
                    << " train accuracy: " << (100.0f * correct / train.images.size()) << "%\n";
        }

        int test_correct = 0;
        for (size_t i = 0; i < test.images.size(); ++i) {
            auto logits = net.forward(test.images[i]);
            int predicted = std::max_element(logits.begin(), logits.end()) - logits.begin();
            if (predicted == test.labels[i]) test_correct++;
        }
        std::cout << "Test accuracy: " << (100.0f * test_correct / test.images.size()) << "%\n";

        net.save("mnist_cnn_weights.bin");
        std::cout << "Weights saved to mnist_cnn_weights.bin\n";
    }
    std::getchar();
    return 0;
}