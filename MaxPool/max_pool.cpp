#include "max_pool.h"
#include <limits>

MaxPool::MaxPool(int channels_, int in_height_, int in_weight_, int pool_size_) : 
                channels(channels_), in_h(in_height_), in_w(in_weight_), pool_size(pool_size_){

    out_h = in_h/pool_size;
    out_w = in_w/pool_size;    
}

std::vector<float> MaxPool::forward(const std::vector<float> &x){

    std::vector<float> out(channels * out_h * out_w, 0.0f);
    max_index.assign(channels*out_h*out_w, -1);

    for(int c = 0; c < channels; c++){
        for(int oy = 0; oy < out_h; oy++){
            for(int ox = 0; ox < out_w; ox++){
                float best = -std::numeric_limits<float>::infinity();
                int best_idx = -1;
                for(int u = 0; u < pool_size; u++){
                    for(int v = 0; v < pool_size; v++){
                            int iy = oy * pool_size + u;
                            int ix = ox * pool_size + v;
                            int idx = in_idx(c, iy, ix);
                            if(x[idx] > best){
                                best = x[idx];
                                best_idx = idx;
                            }
                    }
                }
                int o = out_idx(c, oy, ox);
                out[o] = best;
                max_index[o] = best_idx;
            }
        }
    }
    return out;
};


std::vector<float> MaxPool::backward(const std::vector<float>& grad_out){

    std::vector<float> dx(channels * in_h * in_w, 0.0f);
    for (int o = 0; o < grad_out.size(); o++){
        int src = max_index[o];
        dx[src] += grad_out[o];
    }
    return dx;
}