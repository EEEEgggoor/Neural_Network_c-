#include "relu.h"

std::vector<float> ReLu::forward(const std::vector<float> &x){

    last_input = x;

    // A[i] = max(0, Z[i])
    std::vector<float> out(x.size());
    for(int i = 0; i < x.size(); i++){
        out[i] = x[i] > 0.0f ? x[i] : 0.0f; 
    }

    return out;
}

std::vector<float> ReLu::backward(const std::vector<float> &grad_out){

    //dX[i] = dY[i] если Z[i] > 0, иначе 0
    std::vector<float> dx(grad_out.size(), 0.0f);
    for(int i = 0; i < grad_out.size(); i++){
        dx[i] = last_input[i] > 0.0f ? grad_out[i] : 0;
    }

    return dx;

}