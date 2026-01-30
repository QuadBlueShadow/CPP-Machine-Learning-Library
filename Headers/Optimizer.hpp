#include "Architecture.hpp"

/*
The Adam optimzer takes momentum and RMSProp to create an adaptive learning rate for smoother and faster training
Adam paper: https://arxiv.org/pdf/1412.6980
*/
class Adam{
    public:
        /*
        Model: The model that is to be adjusted by the optimizer
        lr: The learning rate at which we will adjust the model
        */
        Adam(Model* model, float lr);

        /*
        Applies the gradients within the model
        Clip: This is used for clipping gradients within the neural net
        */
        void apply_changes(float clip = 1);
        // Used in case we want to change our learning rate
        void change_lr(float lr);
    private:
        Model* model;
        float lr = 0;

        float beta1 = 0.9;
        float beta2 = 0.999f;

        // Calculates the moments for the weights of the neural net using the current and past gradients
        std::vector<std::vector<float>> calculate_weight_moments(std::vector<std::vector<float>> gradients, std::vector<std::vector<float>> previous_gradients);
        // Calculates the variance for the weights of the neural net using the current gradients and past variance
        std::vector<std::vector<float>> calculate_weight_variance(std::vector<std::vector<float>> gradients, std::vector<std::vector<float>> previous_variance);

        // Calculates the moments for the biases of the neural net using the current and past gradients
        std::vector<float> calculate_bias_moments(std::vector<float> gradients, std::vector<float> previous_gradients);
        // Calculates the variance for the biases of the neural net using the current gradients and past variance
        std::vector<float> calculate_bias_variance(std::vector<float> gradients, std::vector<float> previous_variance);
        
        std::vector<std::vector<std::vector<float>>> previous_weight_gradients;
        std::vector<std::vector<std::vector<float>>> previous_weight_variance;
        std::vector<std::vector<float>> previous_bias_gradients;
        std::vector<std::vector<float>> previous_bias_variance;
};

class BasicOptimizer{
    public:
        /*
        Model: The model that is to be adjusted by the optimizer
        lr: The learning rate at which we will adjust the model
        */
        BasicOptimizer(Model* model, float lr);

        /*
        Applies the gradients within the model
        Clip: This is used for clipping gradients within the neural net
        */
        void apply_changes(float clip = 1);
        // Used in case we want to change our learning rate
        void change_lr(float lr);
    private:
        Model* model;
        float lr = 0;
};