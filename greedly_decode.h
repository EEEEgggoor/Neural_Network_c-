#pragma once
#include <vector>
#include <string>
#include <algorithm>



std::string decode(const std::vector<std::vector<float>>& T){
    const int BLANK = 10;
    std::string res;
    int prev_class = -1;

    for(int t = 0; t < T.size(); t++){
        int best_class = std::max_element(T[t].begin(), T[t].end()) - T[t].begin();
     
        if(best_class!=prev_class){
            if(best_class != BLANK){
                res += std::to_string(best_class); 
            }
        }
        
        prev_class = best_class;
    }

    return res;
}