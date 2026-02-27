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
    layers.push_back(std::make_shared<LeakyRelu>());
    layers.push_back(std::make_shared<LinearLayer>(2, 1));

    Model model(layers);
    L2Loss loss_fn;
    Adam optim(&model, 0.01);

    model.save_net("Net");

    // std::vector<float> targets;

    // for (int i = 0; i < 10; i++){
    //     targets.push_back(i*2 + 1);
    // }

    // std::vector<float> prediction;

    // for (int x = 0; x < 600; x++){
    //     for (int i = 0; i < targets.size(); i++){
    //         prediction = model.run({(float)i});

    //         loss_fn.calculate(prediction, {targets[i]});
    //         model.backprop(loss_fn.derivative());

    //         optim.apply_changes(0.5);
    //     }
    // }

    // for (int i = 0; i < targets.size(); i++){
    //     prediction = model.run({(float)i});

    //     std::cout << "Prediction: " << prediction[0] << ", Target:" << targets[i] << ", Loss:" << loss_fn.calculate(prediction, {targets[i]}) << std::endl;
    //     model.backprop(loss_fn.derivative());

    //     optim.apply_changes(0.5);
    // }
    
    // std::this_thread::sleep_for(std::chrono::seconds(10)); 
    
    return 0;
}