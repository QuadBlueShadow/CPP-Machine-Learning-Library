#include <string>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>

#include "Architecture.hpp"
#include "ActivationFunctions.hpp"
#include "Layers.hpp"
#include "Basic.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

int main() {
    std::vector<std::shared_ptr<NetComponent>> layers;
    layers.push_back(std::make_shared<LinearLayer>(1, 2));
    //layers.push_back(std::make_shared<LeakyRelu>());
    layers.push_back(std::make_shared<LinearLayer>(2, 1));

    Model color_predictor(layers);
    L2Loss loss_fn;
    Adam optim(&color_predictor, 0.01);

    color_predictor.print_net();

    std::vector<float> input = {1};

    std::vector<float> output = color_predictor.run(input);

    std::cout << "Output: " << output[0] << std::endl;

    std::vector<float> desired_output = {10};

    for (int i = 0; i < 1500; i++){
        if (i > 1500)
            optim.change_lr(0.01);

        float loss = loss_fn.calculate(output, desired_output);
        std::vector<float> der = loss_fn.derivative();

        // std::cout << "Loss: " << loss << std::endl;
        //std::cout << "Derivative: " << der[0] << std::endl;
        color_predictor.backprop(der);
        optim.apply_changes(0.5);

        output = color_predictor.run(input);

        std::cout << "Output2: " << output[0] << std::endl;
    }

    //color_predictor.load_net("Net", false);
    color_predictor.print_net();
    
    std::this_thread::sleep_for(std::chrono::seconds(50)); 
    
    return 0;
}