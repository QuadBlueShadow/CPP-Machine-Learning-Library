#include <vector>

// This function calculates our L2 loss from our model (target - prediction) ^ 2
class L2Loss{
    public:
        /*
        Predictions: The prediction values from the model
        Targets: The targets the model should be aiming for
        */
        float calculate(std::vector<float> predictions, std::vector<float> targets);

        // Used in backpropogation to find how we need to adjust the model
        std::vector<float> derivative();
    private:
        // Stores the raw loss values for the derivative function
        std::vector<float> raw_loss;
};