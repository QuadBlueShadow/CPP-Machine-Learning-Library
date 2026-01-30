#pragma once
#include <vector>
#include <memory>

/*
Basic neural net layer class that allow for polymorphism with layers and activation functions.
*/
class NetComponent{
    public:
        //Runs a forward pass through the layer
        virtual std::vector<float> run(std::vector<float> inputs);
        //Runs a backward pass through the layer
        virtual std::vector<float> backprop(std::vector<float> previous);
        virtual void apply_changes(float lr);
        // Used to clip the adjustement gradients to prevent exploding gradients
        virtual void clip_gradients(float clip);

        // Returns the nuerons of a net component
        virtual std::vector<std::vector<float>> get_neurons();
        // Returns the biases of a net component
        virtual std::vector<float> get_biases();
        // Returns the nuerons of a net component
        virtual void set_neurons(int x, int y, float val);
        // Returns the biases of a net component
        virtual void set_biases(int x, float val);

        // Gradient vectors so we can adjust the layer during backpropogation
        std::vector<std::vector<float>> weight_adjustements;
        std::vector<float> bias_adjustements;
        
        // Is this the first layer, used to make sure our biases are correct for the first layer
        bool first = false;
};