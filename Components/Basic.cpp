#include "Basic.hpp"
#include <iostream>

std::vector<float> NetComponent::run(std::vector<float> inputs){
    std::cout << "Uh oh";
    return std::vector<float>(1);
}

std::vector<float> NetComponent::backprop(std::vector<float> previous){
    return std::vector<float>(1);
}

void NetComponent::apply_changes(float lr){
    //std::cout << "Uh oh";
}

void NetComponent::clip_gradients(float clip){
    //std::cout << "Uh oh";
}

std::vector<std::vector<float>> NetComponent::get_neurons(){
    std::vector<std::vector<float>> empty;
    return empty;
}

std::vector<float> NetComponent::get_biases(){
    return std::vector<float>(1);
}