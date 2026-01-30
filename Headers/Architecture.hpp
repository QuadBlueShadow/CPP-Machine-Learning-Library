#pragma once
#include "Basic.hpp"
#include <fstream>
#include <sstream>

// Basic Neural Net
class Model{
    public:
        // Our model's layers
        std::vector<std::shared_ptr<NetComponent>> layers;

        /*
        Layers: The layer architecture of the neural net
        */
        Model(std::vector<std::shared_ptr<NetComponent>> layers);

        // Runs a forward pass
        std::vector<float> run(std::vector<float> inputs);

        /*
        Runs a backward pass and finds the gradients through the neural net
        Loss: The derivative of our loss coming from our loss function
        */
        void backprop(std::vector<float> loss);

        void print_net();

        bool save_net(std::string name);

        bool load_net(std::string name, bool print);
};