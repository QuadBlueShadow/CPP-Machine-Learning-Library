#include "Basic.hpp"
#include <chrono>
#include <random>

/*
This layer passes input through linearly, this is where all the neurons will be held
*/
class LinearLayer : public NetComponent{
    public:
        // Runs a forward pass through the layer
        std::vector<float> run(std::vector<float> inputs) override;
        // Runs a backward pass through the layer
        std::vector<float> backprop(std::vector<float> previous) override;
        // This applies the gradients
        void apply_changes(float lr) override;
        // Used to clip the adjustement gradients to prevent exploding gradients
        void clip_gradients(float clip) override;
        // Returns the nuerons of a net component
        std::vector<std::vector<float>> get_neurons() override;
        // Returns the biases of a net component
        std::vector<float> get_biases() override;

        // Is this the first layer, used to make sure our biases are correct for the first layer
        bool first = false;

        // Gradient vectors so we can adjust the layer during backpropogation
        std::vector<std::vector<float>> weight_adjustements;
        std::vector<float> bias_adjustements;

        /*
        Neurons: Number of neurons
        Outputs: Number of neurons in the next linear layer
        */
        LinearLayer(int neurons, int outputs);
    private:
        // Used for backpropogation
        std::vector<float> last_input;

        // Weights and biases
        std::vector<std::vector<float>> neurons;
        std::vector<float> biases;

        float clamp(float x, float max, float min);
};