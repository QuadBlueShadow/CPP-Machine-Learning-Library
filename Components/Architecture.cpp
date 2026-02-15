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
    for (int i = this->layers.size()-1; i >= 0; i--){
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

bool Model::save_net(std::string name){
    std::ofstream outputFile;

    outputFile.open(name + ".csv");

    if (!outputFile.is_open()) {
        std::cout << "Error: Could not open the file!" << std::endl;
        return false;
    }

    for (int i = 0; i < this->layers.size(); i++){
        outputFile << "Layer," << (i + 1) << std::endl;
        for (int y = 0; y < this->layers[i]->get_neurons().size(); y++){
            outputFile << "Neuron," << (y + 1) << std::endl;
            for (int z = 0; z < this->layers[i]->get_neurons()[y].size(); z++){
                outputFile << "Weight," << (z + 1) << "," << this->layers[i]->get_neurons()[y][z] << std::endl;
            }

            outputFile << "Bias," << (y + 1) << "," << this->layers[i]->get_biases()[y] << std::endl;
            outputFile << std::endl;
        }
    }

    return true;
}

bool Model::load_net(std::string name, bool print){
    std::ifstream file(name + ".csv");

    if (!file.is_open()) {
        std::cout << "Error: Could not open the file." << std::endl;
        return false;
    }

    std::string line;
    int layer = 0;
    int neuron = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> row_data;

        // 3. Parse each line using a comma delimiter
        while (std::getline(ss, field, ',')) {
            row_data.push_back(field);
        }

        // 4. Process/print the extracted data
        if (!row_data.empty()) {
            if (row_data.size() < 3){
                if (print)
                    std::cout << "Type: " << row_data[0] 
                        << ", Num: " << row_data[1] << std::endl;
                
                if (row_data[0] == "Layer"){
                    layer = std::stoi(row_data[1])-1;
                }

                if (row_data[0] == "Neuron"){
                    neuron = std::stoi(row_data[1])-1;
                }

            }else{
                if(print)
                    std::cout << "Type: " << row_data[0] 
                        << ", Num: " << row_data[1] 
                        << ", Value: " << row_data[2] << std::endl;

                if (row_data[0] == "Weight"){
                    layers[layer]->set_neurons(neuron, std::stoi(row_data[1])-1,  std::stof(row_data[2]));
                }

                if (row_data[0] == "Bias"){
                    layers[layer]->set_biases(neuron, std::stof(row_data[2]));
                }
            }
            
        }
    }

    // Close the file
    file.close();

    return true;
}