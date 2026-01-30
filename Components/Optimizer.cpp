#include "Optimizer.hpp"
#include <iostream>

Adam::Adam(Model* model, float lr){
    this->model = model;
    this->lr = lr;

    // Make sure that the vectors that will hold our previous information have numbers within them
    for (int i = 0; i < model->layers.size(); i++){
        previous_weight_gradients.push_back(model->layers[i]->weight_adjustements);
        previous_weight_variance.push_back(model->layers[i]->weight_adjustements);
        previous_bias_gradients.push_back(model->layers[i]->bias_adjustements);
        previous_bias_variance.push_back(model->layers[i]->bias_adjustements);
    }
}

std::vector<std::vector<float>> Adam::calculate_weight_moments(std::vector<std::vector<float>> gradients, std::vector<std::vector<float>> previous_gradients){
    std::vector<std::vector<float>> moments;

    for (int i = 0; i < gradients.size(); i++){
        moments.push_back(std::vector<float>(gradients[i].size()));

        for (int y = 0; i < gradients[i].size(); i++){
            moments[i][y] = beta1 * previous_gradients[i][y] + (1 - beta1) * gradients[i][y];
        }
    }
    
    return moments;
}

std::vector<std::vector<float>> Adam::calculate_weight_variance(std::vector<std::vector<float>> gradients, std::vector<std::vector<float>> previous_variance){
    std::vector<std::vector<float>> variances;

    for (int i = 0; i < gradients.size(); i++){
        variances.push_back(std::vector<float>(gradients[i].size()));

        for (int y = 0; i < gradients[i].size(); i++){
            variances[i][y] = beta2 * previous_variance[i][y] + (1 - beta2) * gradients[i][y] * gradients[i][y];
        }
    }
    
    return variances;
}

std::vector<float> Adam::calculate_bias_moments(std::vector<float> gradients, std::vector<float> previous_gradients){
    std::vector<float> moments(gradients.size());

    for (int i = 0; i < gradients.size(); i++){
        moments[i] = beta1 * previous_gradients[i] + (1 - beta1) * gradients[i];
    }
    
    return moments;
}

std::vector<float> Adam::calculate_bias_variance(std::vector<float> gradients, std::vector<float> previous_variance){
    std::vector<float> variance(gradients.size());

    for (int i = 0; i < gradients.size(); i++){
        variance[i] = beta2 * previous_variance[i] + (1 - beta2) * gradients[i] * gradients[i];
    }
    
    return variance;
}

void Adam::apply_changes(float clip){
    for (int i = 0; i < model->layers.size(); i++){
        // We calculate our data for our weights
        previous_weight_gradients[i] = calculate_weight_moments(model->layers[i]->weight_adjustements, previous_weight_gradients[i]);
        previous_weight_variance[i] = calculate_weight_variance(model->layers[i]->weight_adjustements, previous_weight_variance[i]);

        // We apply the momnents and variance adjusting for bias into each one of our neural net's gradients
        for (int x = 0; x < model->layers[i]->weight_adjustements.size(); x++){
            for (int y = 0; y < model->layers[i]->weight_adjustements[0].size(); y++){
                model->layers[i]->weight_adjustements[x][y] = previous_weight_gradients[i][x][y] / (previous_weight_variance[i][x][y] + 1e-10f);
            }
        }

        // We calculate our data for our biases
        previous_bias_gradients[i] = calculate_bias_moments(model->layers[i]->bias_adjustements, previous_bias_gradients[i]);
        previous_bias_variance[i] = calculate_bias_variance(model->layers[i]->bias_adjustements, previous_bias_variance[i]);

        // We apply the momnents and variance adjusting for bias into each one of our neural net's gradients
        for (int y = 0; y < model->layers[i]->bias_adjustements.size(); y++){
            model->layers[i]->bias_adjustements[y] = previous_bias_gradients[i][y] / (previous_bias_variance[i][y] + 1e-8f);
        }

        // We make sure to clip the gradients
        model->layers[i]->clip_gradients(clip);

        // We apply the changes on the neural net using the learing rate
        model->layers[i]->apply_changes(lr);
    }
}

void Adam::change_lr(float lr){
    this->lr = lr;
}

// ------------------------------------------------------------

BasicOptimizer::BasicOptimizer(Model* model, float lr){
    this->model = model;
    this->lr = lr;
}

void BasicOptimizer::apply_changes(float clip){
    // We go through each layer and clip the gradients and then adjust the gradients
    for (int i = 0; i < model->layers.size(); i++){
        model->layers[i]->clip_gradients(clip);

        model->layers[i]->apply_changes(lr);
    }
}

void BasicOptimizer::change_lr(float lr){
    this->lr = lr;
}