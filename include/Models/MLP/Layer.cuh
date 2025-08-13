#pragma once

#include "../../DeviceMatrix.cuh"
#include "../../Matrix.hpp"
#include <utility> // for std::pair
#include <cstddef> // for size_t
#include "../LearningUnit.cuh"


namespace dl {

    namespace MLP {

        class Layer : public LearningUnit {
        protected:
        	DeviceMatrix weights;
        	DeviceMatrix biases;
        	DeviceMatrix input;
        
        	//size_t input_size;
        	//size_t output_size;
        
        public:
        	Layer(const DeviceMatrix& Weights, const DeviceMatrix& Biases);
        	Layer(DeviceMatrix&& Weights, DeviceMatrix&& Biases);
        	Layer();
        
        	Layer(const Layer& layer) = default;
        	Layer& operator=(const Layer& layer) = default;
        
        	Layer(Layer&& layer) noexcept = default;
        	Layer& operator=(Layer&& layer) noexcept = default;
        
        	void setInput(DeviceMatrix&& input_matrix);
        	void setInput(const DeviceMatrix& input_matrix);
        
        	void clean() noexcept;
        
        
        	// Forward pass and backpropagation
        
        	DeviceMatrix forward() override; 
        	DeviceMatrix backpropagate(DeviceMatrix grad_output) override;
        	DeviceMatrix updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate = 0.01f) override;
        
        
        	size_t inputSize() const override { return weights.cols(); }
        	size_t outputSize() const override { return weights.rows(); }
			size_t parameterCount() const override { return weights.totalSize() + biases.totalSize(); }
        
        	static Layer RandomLayer(size_t input_size, size_t output_size, std::pair<float, float> range = { 0.0f, 1.0f });

            void print(const char* header = "\nMLP::Layer analytics:") const;

			bool has_variable_input() const override { return false; }  // is input size fixed of variable for layer, if its fixed, it need to be set before training

            std::shared_ptr<LearningUnit> clone() const override;

            virtual ~Layer() = default;

		}; // class Layer
	}; // namespace MLP
};  // namespace dl