#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include "softmax_ce.h"

#define CTC_NUM_CLASSES 11
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

std::vector<std::vector<float>> softmax_T(std::vector<std::vector<float>> &T){

    int num_step = T.size();

    std::vector<std::vector<float>> out(num_step, std::vector<float>(CTC_NUM_CLASSES));
    std::vector<float> vector_in_T(CTC_NUM_CLASSES);

    for(int i = 0; i < num_step; i++){
        vector_in_T.assign(CTC_NUM_CLASSES, 0.0f);
        for(int j = 0; j < CTC_NUM_CLASSES; j++){
            vector_in_T[j] = T[i][j];
        }
        out[i] = softmax(vector_in_T);
    }
    return out;
}





float log_sum_exp(float log_a, float log_b) {
    if (log_a == NEG_INF) return log_b;
    if (log_b == NEG_INF) return log_a;
    float m = std::max(log_a, log_b);
    return m + std::log(std::exp(log_a - m) + std::exp(log_b - m));
}

float ctc_loss_and_grad(const std::vector<std::vector<float>>& y, const std::vector<int>& label_seq, std::vector<std::vector<float>>& grad_out){

    int S = 2*label_seq.size() + 1;
    int T = y.size();
    int BLANK = 10;

    std::vector<int> label_seq_extend(S, 0);

    for(int s = 0; s < S; s++){
        if(s%2==0) { label_seq_extend[s] = BLANK; }
        else { label_seq_extend[s] = label_seq[(s)/2]; }
    }


    std::vector<std::vector<float>> log_y(T, std::vector<float>(CTC_NUM_CLASSES));
    for (int t = 0; t < T; t++)
        for (int k = 0; k < CTC_NUM_CLASSES; k++)
            log_y[t][k] = std::log(y[t][k] + 1e-9f);

    std::vector<std::vector<float>> a(T, std::vector<float>(S, NEG_INF));  

    //ALPHA
    //t = 0
    a[0][0] = log_y[0][label_seq_extend[0]];
    if (S > 1) a[0][1] = log_y[0][label_seq_extend[1]];


    for (int t = 1; t < T; t++){
        for (int s = 0; s < S; s++){
            float _a = a[t-1][s];
            if (s - 1 >= 0) _a = log_sum_exp(_a, a[t-1][s-1]);

            bool can_skip = (label_seq_extend[s] != BLANK) && (s >= 2) && (label_seq_extend[s] != label_seq_extend[s-2]);
            if (can_skip) _a = log_sum_exp(_a, a[t-1][s-2]);

            a[t][s] = _a + log_y[t][label_seq_extend[s]];  
        }
    }

    // Z = a[T-1][S-1] + a[T-1][S-2])
    float logZ = (S > 1) ? log_sum_exp(a[T-1][S-1], a[T-1][S-2]) : a[T-1][S-1];


    //BETA
    std::vector<std::vector<float>> b(T, std::vector<float>(S, NEG_INF));
    b[T-1][S-1] = 0.0f;  
    if (S > 1) b[T-1][S-2] = 0.0f;


    for (int t = T - 2; t>=0; t--){
        for (int s = S - 1; s>=0; s--){
            float _b = b[t+1][s];
            if (s+1 < S) _b = log_sum_exp(_b, b[t+1][s+1]);

            bool can_skip = (label_seq_extend[s] != BLANK) && (s + 2 < S) && (label_seq_extend[s] != label_seq_extend[s+2]);
            if (can_skip) _b = log_sum_exp(_b, b[t+1][s+2]);

            b[t][s] = _b + log_y[t][label_seq_extend[s]];
        }
    }


    grad_out.assign(T, std::vector<float>(CTC_NUM_CLASSES, 0.0f));

    for (int t = 0; t < T; t++){

        std::vector<float> log_S_k(CTC_NUM_CLASSES, NEG_INF);

        for (int s = 0; s < S; s++){
            int k = label_seq_extend[s];
            float log_ab = a[t][s] + b[t][s];       // log(alpha*beta) = log(alpha)+log(beta)
            log_S_k[k] = log_sum_exp(log_S_k[k], log_ab);
        }

        for (int k = 0; k < CTC_NUM_CLASSES; k++){
            if (log_S_k[k] == NEG_INF) {

                grad_out[t][k] = y[t][k];
            } else {
                //S_k[k] / (Z * y[t][k])  ->  exp(log_S_k[k] - logZ - log(y[t][k]))
                float exponent = log_S_k[k] - logZ - log_y[t][k];
                exponent = std::min(exponent, 30.0f);
                float ratio = std::exp(exponent);
                grad_out[t][k] = y[t][k] - ratio;
            }
        }
    }

    float Loss = -logZ;

    return Loss;
}