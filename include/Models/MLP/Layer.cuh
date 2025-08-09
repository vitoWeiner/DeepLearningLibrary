#pragma once

#include "../../DeviceMatrix.cuh"
#include "../../Matrix.hpp"

namespace MLP {


    class Layer {
    protected:
    	DeviceMatrix weights;
    	DeviceMatrix biases;
    	DeviceMatrix input;
    
    	size_t input_size;
    	size_t output_size;
    
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
    
    	DeviceMatrix forward(); 
    	DeviceMatrix backpropagate(const DeviceMatrix& grad_output);
    	DeviceMatrix train(const DeviceMatrix& gradient_output, float learning_rate = 0.01f);
    
    
    	size_t inputSize() const { return input_size; }
    	size_t outputSize() const { return output_size; }
    
    	static Layer RandomLayer(size_t input_size, size_t output_size, std::pair<float, float> range = { 0.0f, 1.0f });
    };

};