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
        
        	Layer(const Layer& layer);
        	Layer& operator=(const Layer& layer);
        
        	Layer(Layer&& layer) noexcept;
        	Layer& operator=(Layer&& layer) noexcept;
        
        	void setInput(DeviceMatrix&& input_matrix);
        	void setInput(const DeviceMatrix& input_matrix);
        
        	void clean() noexcept;
        
        
        	// Forward pass and backpropagation
        
        	DeviceMatrix forward() override; 
        	DeviceMatrix backpropagate(const DeviceMatrix& grad_output) override;
        	DeviceMatrix train(const DeviceMatrix& gradient_output, float learning_rate = 0.01f) override;
        
        
        	size_t inputSize() const override { return weights.cols(); }
        	size_t outputSize() const override { return weights.rows(); }
        
        	static Layer RandomLayer(size_t input_size, size_t output_size, std::pair<float, float> range = { 0.0f, 1.0f });

            void print(const char* header = "\nMLP::Layer analytics:") const;

            std::unique_ptr<LearningUnit> clone() const override;

		}; // class Layer
	}; // namespace MLP
};  // namespace dl