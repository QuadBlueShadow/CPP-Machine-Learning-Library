#include "Loss.hpp"

float L2Loss::calculate(std::vector<float> predictions, std::vector<float> targets){
    float output = 0; // This will be our overall L2 loss
    raw_loss = {}; // We store this to make sure that we can use our raw loss values for back propogation

    for (int i = 0; i < predictions.size(); i++){
        // L2 loss (target - prediction) ^ 2
        raw_loss.push_back(targets[i] - predictions[i]);
        output += raw_loss[i] * raw_loss[i];
    }

    // Get the average of our loss
    output /= raw_loss.size();

    return output;
}

std::vector<float> L2Loss::derivative(){
    // Our derivative of L2 loss
    // (target - prediction) ^ 2 => 2 * (target - prediction)
    for (int i = 0; i < raw_loss.size(); i++){
        raw_loss[i] *= 2;
    }

    return raw_loss;
}