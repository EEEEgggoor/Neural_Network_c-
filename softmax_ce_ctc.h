#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include "softmax_ce.h"

#define CTC_NUM_CLASSES 11


constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

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


double log_sum_exp(double log_a, double log_b) {
    if (log_a == NEG_INF) return log_b;
    if (log_b == NEG_INF) return log_a;
    double m = std::max(log_a, log_b);
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



    std::vector<std::vector<double>> log_y(T, std::vector<double>(CTC_NUM_CLASSES));
    for (int t = 0; t < T; t++)
        for (int k = 0; k < CTC_NUM_CLASSES; k++)
            log_y[t][k] = std::log((double)y[t][k] + 1e-9);

    std::vector<std::vector<double>> a(T, std::vector<double>(S, NEG_INF));

    //ALPHA
    //t = 0
    a[0][0] = log_y[0][label_seq_extend[0]];
    if (S > 1) a[0][1] = log_y[0][label_seq_extend[1]];


    for (int t = 1; t < T; t++){
        for (int s = 0; s < S; s++){
            double _a = a[t-1][s];
            if (s - 1 >= 0) _a = log_sum_exp(_a, a[t-1][s-1]);

            bool can_skip = (label_seq_extend[s] != BLANK) && (s >= 2) && (label_seq_extend[s] != label_seq_extend[s-2]);
            if (can_skip) _a = log_sum_exp(_a, a[t-1][s-2]);

            a[t][s] = _a + log_y[t][label_seq_extend[s]];
        }
    }


    // Z = a[T-1][S-1] + a[T-1][S-2])
    double logZ = (S > 1) ? log_sum_exp(a[T-1][S-1], a[T-1][S-2]) : a[T-1][S-1];
    if (logZ == NEG_INF || !std::isfinite(logZ)) {
        grad_out.assign(T, std::vector<float>(CTC_NUM_CLASSES, 0.0f));
        return 0.0f;
    }


    //BETA
    std::vector<std::vector<double>> b(T, std::vector<double>(S, NEG_INF));
    b[T-1][S-1] = 0.0;
    if (S > 1) b[T-1][S-2] = 0.0;

    for (int t = T - 2; t >= 0; t--) {
        for (int s = S - 1; s >= 0; s--) {
            double _b = b[t+1][s]; // переход в тот же символ (blank или повтор)

            if (s + 1 < S) _b = log_sum_exp(_b, b[t+1][s+1] + log_y[t+1][label_seq_extend[s+1]]);

            bool can_skip = (label_seq_extend[s] != BLANK) && (s + 2 < S) && (label_seq_extend[s] != label_seq_extend[s+2]);
            if (can_skip) _b = log_sum_exp(_b, b[t+1][s+2] + log_y[t+1][label_seq_extend[s+2]]);

            b[t][s] = _b;
        }
    }


    grad_out.assign(T, std::vector<float>(CTC_NUM_CLASSES, 0.0f));

    for (int t = 0; t < T; t++){

        std::vector<double> log_S_k(CTC_NUM_CLASSES, NEG_INF);

        for (int s = 0; s < S; s++){
            int k = label_seq_extend[s];
            double log_ab = a[t][s] + b[t][s];       // log(alpha*beta) = log(alpha)+log(beta)
            log_S_k[k] = log_sum_exp(log_S_k[k], log_ab);
        }

        for (int k = 0; k < CTC_NUM_CLASSES; k++){
            if (log_S_k[k] == NEG_INF) {
                grad_out[t][k] = y[t][k];
            } else {
                // S_k[k] / Z  ->  exp(log_S_k[k] - logZ)
                double exponent = log_S_k[k] - logZ;
                exponent = std::min(exponent, 30.0);
                double ratio = std::exp(exponent);
                grad_out[t][k] = (float)((double)y[t][k] - ratio);
            }
        }
    }

    double Loss = -logZ;

    return (float)Loss;
}