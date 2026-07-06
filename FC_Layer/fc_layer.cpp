#include "fc_layer.h"
#include <cmath>
#include <random>
#include <vector>

FC_Layer::FC_Layer(int in_feat_, int out_feat_) : in_feat(in_feat_), out_feat(out_feat_) {
    W.assign(out_feat, std::vector<float>(in_feat));
    b.assign(out_feat, 0.0f);

    std::mt19937 gen(42);
    float limit = std::sqrt(6.0f / (out_feat + in_feat));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (auto &row : W){
        for(auto &w : row){
            w = dist(gen);
        }
    }    
};


std::vector<float> FC_Layer::forward(const std::vector<float> &x) {
    last_input = x;

    //z_i = b_i + Σ(j=0..399) W[i][j]·x_j,   i=0..63
    std::vector<float> out(out_feat, 0.0f);
    for(int i = 0; i < out_feat; i++){
        float sum = b[i];
        for(int j = 0; j < in_feat; j++){
            sum += W[i][j]*x[j];
        }
        out[i] = sum;
    }
    return out;
}

std::vector<float> FC_Layer::backward(const std::vector<float> &dz){
    dW.assign(out_feat, std::vector<float>(in_feat));

    //dW[i][j] = dz_i·x_j
    for(int i = 0; i < out_feat; i++){
        for(int j = 0; j < in_feat; j++){
            dW[i][j] = dz[i] * last_input[j];
        }
    }

    //db_i = dz_iW
    db = dz;


    //dx_j = Σ(i=0..63) W[i][j]·dz_i
    std::vector<float> dx(in_feat, 0.0f);
    for(int i=0; i < out_feat; i++){
        for(int j = 0; j < in_feat; j++){
            dx[j] += W[i][j] * dz[i];
        }
    }
    return dx;
}


std::vector<std::vector<float>> FC_Layer::forward_T(const std::vector<float> &T){

    int num_step = T.size()/in_feat;

    last_inputs_T.assign(num_step, std::vector<float>(in_feat));

    std::vector<std::vector<float>> out(num_step, std::vector<float>(out_feat, 0.0f));
    
    for(int t = 0; t < num_step; t++){

        //прямой проход каждого FC слоя 
        for(int j = 0; j < in_feat; j++){
            last_inputs_T[t][j] = T[t*in_feat + j];
        }

        for(int i = 0; i < out_feat; i++){
            float sum = b[i];
            for(int j = 0; j < in_feat; j++){
                sum += W[i][j]*last_inputs_T[t][j];
            }
            out[t][i] = sum;
        }
    }
    return out;
}


std::vector<float> FC_Layer::backward_T(const std::vector<std::vector<float>>& dz){

    int num_step = dz.size();

    dW.assign(out_feat, std::vector<float>(in_feat));
    db.assign(out_feat, 0.0f); 

    std::vector<float> dx_T(num_step * in_feat, 0.0f);


    for(int t = 0; t < num_step; t++){

        //обратный проход каждого FC слоя 
        for(int i = 0; i < out_feat; i++){
            db[i] += dz[t][i];
            for (int j = 0; j < in_feat; j++){
                dW[i][j] += dz[t][i] * last_inputs_T[t][j];
                dx_T[t * in_feat + j] += W[i][j] * dz[t][i];
            }
        }
    }
    return dx_T;
}

void FC_Layer::update(float lr){
    //градиентные спуск
    for(int i = 0; i < out_feat; i++){
        for(int j = 0; j < in_feat; j++){
            W[i][j] -= lr*dW[i][j];
        }
        b[i] -= lr*db[i];
    }
}

void FC_Layer::save_w(std::ofstream& out) {

    for (int i = 0; i < out_feat; ++i)
        out.write(reinterpret_cast<const char*>(W[i].data()), in_feat * sizeof(float));
    out.write(reinterpret_cast<const char*>(b.data()), out_feat * sizeof(float));
}

void FC_Layer::load_w(std::ifstream& in) {
    for (int i = 0; i < out_feat; ++i)
        in.read(reinterpret_cast<char*>(W[i].data()), in_feat * sizeof(float));
    in.read(reinterpret_cast<char*>(b.data()), out_feat * sizeof(float));
}

double FC_Layer::grad_squared_norm() const {
    double s = 0.0;
    for (const auto& row : dW)
        for (float v : row)
            s += (double)v * v;
    for (float v : db) s += (double)v * v;
    return s;
}

void FC_Layer::scale_grad(float scale) {
    for (auto& row : dW)
        for (float& v : row)
            v *= scale;
    for (float& v : db) v *= scale;
}