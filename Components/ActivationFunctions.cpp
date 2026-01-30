#include "ActivationFunctions.hpp"

std::vector<float> LeakyRelu::run(std::vector<float> inputs){
    // Run through each part of the input and decide wether to divide by ten or not
    // We divide by ten if the number is negative
    for(int i = 0; i < inputs.size(); i++){
        inputs[i] = (inputs[i] > 0) ? inputs[i] : inputs[i] * 0.1;
    }

    return inputs;
}

std::vector<float> LeakyRelu::backprop(std::vector<float> previous){
    // In the backwards pass we do the same thing as we do for the forward pass as the derivatives of leaky relu
    // are just the coefficients 1 and 1/10 based on if the number is positive or negative
    for(int i = 0; i < previous.size(); i++){
        previous[i] = (previous[i] > 0) ? previous[i] : previous[i] * 0.1;
    }

    return previous;
}