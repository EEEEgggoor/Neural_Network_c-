#include "pooling.h"


Pooling::Pooling(int channels_, int in_h_, int in_w_) : 
         channels(channels_), in_h(in_h_), in_w(in_w_) {
        out_h = in_h;
        out_w = in_w;
};

std::vector<float> Pooling::forward_average_pooling(const std::vector<float> &x){

    std::vector out(channels * in_w, 0.0f);

    for (int c = 0; c < channels; c++){
        for (int w = 0; w < in_w; w++){
            float sum = 0.0f;
            for(int h = 0; h < in_h; h++){
                sum += x[in_idx(c, h, w)];
            }
            out[c * in_w + w] = sum  / static_cast<float>(in_h);
        }
    }
    out = transpose_C_W_to_W_C(out);
    return out;
};

std::vector<float> Pooling::backward_average_pooling(const std::vector<float> &grad_out){

    std::vector<float> dX(channels * in_h * in_w);

    std::vector<float> grad_out_transp(channels * in_w);
    grad_out_transp = transpose_W_C_to_C_W(grad_out);

    for(int c = 0; c < channels; c++){
        for(int h = 0; h < in_h; h++){
            for(int w = 0; w < in_w; w++){
                dX[out_idx(c, h, w)] = grad_out_transp[c * in_w + w] / static_cast<float>(in_h);
            }
        }
    }
    return dX;
}

std::vector<float> Pooling::transpose_C_W_to_W_C(const std::vector<float> _pooling){
    std::vector<float> T(channels * in_w, 0.0f);
    for(int w = 0; w < in_w; w++){
        for(int c = 0; c < channels; c++){
            T[w * channels + c] = _pooling[c * in_w + w];
        } 
    }

    return T;
}


std::vector<float> Pooling::transpose_W_C_to_C_W(const std::vector<float> _pooling){
    std::vector<float> T(channels * in_w, 0.0f);
    for(int w = 0; w < in_w; w++){
        for(int c = 0; c < channels; c++){
            T[c * in_w + w] = _pooling[w * channels + c];
        } 
    }

    return T;
}

std::vector<float> Pooling::forward(const std::vector<float> &x){
    return forward_average_pooling(x);
}
 
std::vector<float> Pooling::backward(const std::vector<float> &grad_out){
    return backward_average_pooling(grad_out);
}