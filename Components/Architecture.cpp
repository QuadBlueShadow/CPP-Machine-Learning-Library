#include "Architecture.hpp"
#include <iostream>

Model::Model(std::vector<std::shared_ptr<NetComponent>> layers){
    this->layers = layers;
    layers[0]->first = true;
}

std::vector<float> Model::run(std::vector<float> inputs){
    // Go through each layer and eventually return the final output
    for (int i = 0; i < this->layers.size(); i++){
        inputs = layers[i]->run(inputs);
    }

    return inputs;
}

void Model::backprop(std::vector<float> loss){
    // Go through each layer and run backpropogation to find our adjustement gradients
    for (int i = 0; i < this->layers.size(); i++){
        loss = layers[i]->backprop(loss);
    }
}

void Model::print_net(){
    for (int i = 0; i < this->layers.size(); i++){
        std::cout << "Layer " << (i + 1) << ":" << std::endl;

        for (int y = 0; y < this->layers[i]->get_neurons().size(); y++){
            std::cout << "    Neuron " << (y + 1) << ":" << std::endl;

            for (int z = 0; z < this->layers[i]->get_neurons()[y].size(); z++){
                std::cout << "       Weight " << (z + 1) << ": " << this->layers[i]->get_neurons()[y][z] << std::endl;
            }

            std::cout << "    Bias " << (y + 1) << ": " << this->layers[i]->get_biases()[y] << std::endl;
            std::cout << std::endl;
        }
    }
}