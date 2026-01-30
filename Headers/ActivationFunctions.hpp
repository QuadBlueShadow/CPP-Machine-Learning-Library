#include "Basic.hpp"

/*
Implements the Leaky Relu function which allows full positive activation and 1/10 negative activation
*/
class LeakyRelu : public NetComponent{
    public:
        //Runs a forward pass through the layer
        std::vector<float> run(std::vector<float> inputs) override;
        //Runs a backward pass through the layer
        std::vector<float> backprop(std::vector<float> previous) override;
};