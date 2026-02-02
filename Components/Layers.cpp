#include "Layers.hpp"
#include <iostream>

LinearLayer::LinearLayer(int neurons, int outputs){
    // Used to randomly set our nerual net
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distr(-100, 100);

    // We create new neurons
    for (int i = 0; i < neurons; i++){
        std::vector<float> new_neuron;
        weight_adjustements.push_back(std::vector<float>(outputs));

        // Fill every weight with a random value
        for (int y = 0; y < outputs; y++){
            // We divide by 50 as that gives us the range [-2, 2]
            new_neuron.push_back(distr(gen)/50.0);
        }

        // Append neuron to the vector of neurons
        this->neurons.push_back(new_neuron);
    }

    //std::cout << this->neurons.size();

    // Randomly generate the biases
    for (int y = 0; y < neurons; y++){
        this->biases.push_back(distr(gen)/50.0);
        this->bias_adjustements.push_back(0);
    }
    
}

std::vector<float> LinearLayer::run(std::vector<float> inputs){
    // Create our output vector with the size matching the number of neurons in the next layer
    std::vector<float> outputs(neurons[0].size());

    // We add the biases to our input
    if (!first){
        for (int i = 0; i < biases.size(); i++){
            inputs[i] += biases[i];
        }
    }
   
    last_input = inputs;

    // We go through each neuron
    for (int i = 0;  i < inputs.size(); i++){
        // Within each neuron we look at each weight and multiply it by the input of that neuron
        // We then add that to the correct part of the output
        for (int y = 0; y < neurons[i].size(); y++){
            outputs[y] += neurons[i][y] * inputs[i];
        }
    }

    return outputs;
}

std::vector<float> LinearLayer::backprop(std::vector<float> previous){
    // This is we send to the layer before this one
    std::vector<float> outputs(neurons.size());

    for (int i = 0; i < neurons.size(); i++){
        for (int y = 0; y < neurons[i].size(); y++){
            // We go through each neuron and determine the addition off all of the gradients affecting its weights 
            // which we then send out of the function
            outputs[i] += neurons[i][y] * previous[y];

            // Here we save the adjustements to each individual weight and bias for the optimizer to use later
            weight_adjustements[i][y] = last_input[i] * previous[y];
            bias_adjustements[i] += previous[y];
        }
    }

    return outputs;
}

void LinearLayer::apply_changes(float lr){
    //std::cout << "APC" << std::endl;
    // Here we go through each nueron and apply the adjustements multiplied by a learning rate
    for (int i = 0; i < neurons.size(); i++){
        for (int y = 0; y < neurons[i].size(); y++){
            //std::cout << "Initial: " << neurons[i][y] << std::endl;
            neurons[i][y] += weight_adjustements[i][y] * lr;
            //std::cout << "After: " << neurons[i][y] << std::endl;
        }
    }

    if (!first){
        // Here we go through each bias and apply the adjustements multiplied by a learning rate
        for (int i = 0; i < bias_adjustements.size(); i++){
            //std::cout << "Initial: " << biases[i] << std::endl;
            biases[i] += bias_adjustements[i] * lr;
            bias_adjustements[i] = 0;
        }
    }
}

float LinearLayer::clamp(float x, float max, float min){
    // Clamp an input value (x) to a max and min value
    if (x > max)
        x = max;
    else if (x < min)
        x = min;

    return x;
}

void LinearLayer::clip_gradients(float clip){
    // Here we go through each nueron and apply the adjustements multiplied by a learning rate
    for (int i = 0; i < neurons.size(); i++){
        for (int y = 0; y < neurons[i].size(); y++){
            //std::cout << "Initial: " << neurons[i][y] << std::endl;
            weight_adjustements[i][y] = clamp(weight_adjustements[i][y], clip, -clip);
            //std::cout << "After: " << neurons[i][y] << std::endl;
        }
    }

    if (!first){
        // Here we go through each bias and apply the adjustements multiplied by a learning rate
        for (int i = 0; i < bias_adjustements.size(); i++){
            bias_adjustements[i] = clamp(bias_adjustements[i], clip, -clip);
        }
    }
}

std::vector<std::vector<float>> LinearLayer::get_neurons(){
    return this->neurons;
}

std::vector<float> LinearLayer::get_biases(){
    return this->biases;
}

void LinearLayer::set_neurons(int x, int y, float val){
    neurons[x][y] = val;
}

void LinearLayer::set_biases(int x, float val){
    biases[x] = val;
}