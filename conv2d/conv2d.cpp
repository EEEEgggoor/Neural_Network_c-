#include "conv2d.h"
#include <cmath>
#include <random>
#include <vector>



Conv2d::Conv2d(int in_channels_, int in_height_, int in_weight_, int count_kernel_, int size_kernel_):
    in_channels(in_channels_), in_h(in_height_), in_w(in_weight_), count_kernel(count_kernel_), size_kernel(size_kernel_){

    out_h = in_h - size_kernel + 1;
    out_w = in_w - size_kernel + 1;

    //K[ck][ic][u][v]
    K.assign(count_kernel, std::vector<std::vector<std::vector<float>>>(in_channels, std::vector<std::vector<float>>(size_kernel, std::vector<float>(size_kernel))));
    bias.assign(count_kernel, 0.0f);

    std::mt19937 gen(42);
    int fan_in = in_channels * size_kernel * size_kernel;
    int fan_out = count_kernel * size_kernel * size_kernel;
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for(auto &ck : K){
        for(auto &ic : ck){
            for(auto &r : ic){
                for(auto &w : r){
                    w = dist(gen);
                }
            }
        }
    }
}


std::vector<float> Conv2d::forward(const std::vector<float> &x){

    last_input = x;
    std::vector<float> out(count_kernel * out_w * out_h, 0.0f);

    for(int ck = 0; ck < count_kernel; ck++){
        for(int oy = 0; oy < out_h; oy++){
            for(int ox = 0; ox < out_w; ox++){
                float sum = bias[ck];
                for(int ic = 0; ic < in_channels; ic++){
                    for(int u = 0; u < size_kernel; u++){
                        for(int v = 0; v < size_kernel; v++){
                            sum += K[ck][ic][u][v] * x[in_idx(ic, oy+u, ox+v)];
                        }
                    }
                }
                out[out_idx(ck, oy, ox)] = sum;
            }
        }
    }
    return out;
}


std::vector<float> Conv2d::backward(const std::vector<float>& grad_out){
    dK.assign(count_kernel, std::vector<std::vector<std::vector<float>>>(in_channels, std::vector<std::vector<float>>(size_kernel, std::vector<float>(size_kernel)))); 
    dbias.assign(count_kernel, 0.0f);
    std::vector<float> dX(in_channels * in_h * in_w, 0.0f);
    
    
    for(int ck = 0; ck < count_kernel; ck++){
        for(int oy = 0; oy < out_h; oy++){
            for(int ox = 0; ox < out_w; ox++){
                dbias[ck] += grad_out[out_idx(ck, oy, ox)];
                for(int ic = 0; ic < in_channels; ic++){
                    for(int u = 0; u < size_kernel; u++){
                        for(int v = 0; v < size_kernel; v++){
                            dK[ck][ic][u][v] += grad_out[out_idx(ck, oy, ox)] * last_input[in_idx(ic, oy+u, ox+v)];
                            dX[in_idx(ic, oy+u, ox+v)] += K[ck][ic][u][v] * grad_out[out_idx(ck, oy, ox)];
                        }
                    }
                }
            }
        }
    }

    return dX;
}

void Conv2d::update(float lr){
    for(int ck = 0; ck < count_kernel; ck++){
        for(int ic = 0; ic < in_channels; ic++){
            for(int u = 0; u < size_kernel; u++){
                for(int v = 0; v < size_kernel; v++){
                    K[ck][ic][u][v] -= lr*dK[ck][ic][u][v];
                }
            }
        }
        bias[ck] -= lr*dbias[ck];
    }
}
void Conv2d::save_w(std::ofstream& out) {
    // Каждая самая внутренняя строка K[oc][ic][u] пишется одним вызовом write
    for(int ck = 0; ck < count_kernel; ck++)
        for (int ic = 0; ic < in_channels; ic++)
            for (int u = 0; u < size_kernel; u++)
                out.write(reinterpret_cast<const char*>(K[ck][ic][u].data()), size_kernel * sizeof(float));
    out.write(reinterpret_cast<const char*>(bias.data()), count_kernel * sizeof(float));
}

void Conv2d::load_w(std::ifstream& in) {
    for(int ck = 0; ck < count_kernel; ck++)
        for (int ic = 0; ic < in_channels; ic++)
            for (int u = 0; u < size_kernel; u++)
                in.read(reinterpret_cast<char*>(K[ck][ic][u].data()), size_kernel * sizeof(float));
    in.read(reinterpret_cast<char*>(bias.data()), count_kernel * sizeof(float));
}