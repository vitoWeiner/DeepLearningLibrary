#pragma once

#include "../../DeviceMatrix.cuh"
#include "../LearningUnit.cuh"

namespace dl {

    class Sigmoid : public  LearningUnit {

    protected:
          DeviceMatrix output;
    
    public:
        Sigmoid();

		DeviceMatrix forward() override;

		DeviceMatrix backpropagate(DeviceMatrix grad_output) override;

		DeviceMatrix updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate = 0.01f) override;


        size_t parameterCount() const override { return 0; }

        void print(const char* header = "\nLearningUnit analytics:") const override;

        bool has_variable_input() const override { return true; }

        bool backpropNeedsInput() const override { return false; }
    
    
    };
};