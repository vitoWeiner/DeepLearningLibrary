#include "../../../include/Models/MLP/Layer.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility> // for std::pair
#include <cstddef> // for size_t
#include <memory>


namespace dl {

	namespace MLP {

        Layer::Layer() {}
        
        Layer::Layer(const DeviceMatrix& Weights, const DeviceMatrix& Biases) {
        
        	if (Weights.rows() != Biases.rows())
        		throw std::runtime_error("Weights and Biases must have the same number of rows");
        
        	this->weights = Weights;
        	this->biases = Biases;
        	//this->input_size = Weights.cols();
        	//this->output_size = Weights.rows();
        
        }
        
        
        
        Layer::Layer(DeviceMatrix&& Weights, DeviceMatrix&& Biases) {
        
        	if (Weights.rows() != Biases.rows())
        		throw std::runtime_error("Weights and Biases must have the same number of rows");
        
        	this->weights = std::move(Weights);
        	this->biases = std::move(Biases);
        	//this->input_size = this->weights.cols();
        	//this->output_size = this->weights.rows();
        }
        
        /*Layer::Layer(const Layer& layer) :
        	weights(layer.weights),
        	biases(layer.biases),
        	input(layer.input)
            {}*/
        	//input_size(layer.input_size),
        	//output_size(layer.output_size) 
        
        
        
        /*Layer& Layer::operator=(const Layer& layer) {
        	if (this == &layer) {
        		return *this;
        	}
        	this->weights = layer.weights;
        	this->biases = layer.biases;
        	//this->input_size = layer.input_size;
        	//this->output_size = layer.output_size;
        	return *this;
        }*/
        
        
        void Layer::setInput(const DeviceMatrix& input_matrix) {
        
        	if (input_matrix.rows() != this->inputSize())
        		throw std::runtime_error("Input matrix must have the same number of rows as the layer's input size");
        
        	this->input = input_matrix;
        }
        
        void Layer::setInput(DeviceMatrix&& input_matrix) {
        	if (input_matrix.rows() != this->inputSize())
        		throw std::runtime_error("Input matrix must have the same number of rows as the layer's input size");
        
        	this->input = std::move(input_matrix);
        }
        
        
        /*Layer::Layer(Layer&& layer) noexcept :
        	weights(std::move(layer.weights)),
        	biases(std::move(layer.biases)),
        	input(std::move(layer.input))
        {
        	//layer.input_size = 0;
        	//layer.output_size = 0;
        }*/
        	//input_size(layer.input_size),
        	//output_size(layer.output_size) 
        
        
       /* Layer& Layer::operator=(Layer&& layer) noexcept {
        	if (this == &layer) {
        		return *this;
        	}
        	this->weights = std::move(layer.weights);
        	this->biases = std::move(layer.biases);
        	this->input = std::move(layer.input);
        	//this->input_size = layer.input_size;
        	//this->output_size = layer.output_size;
        	//layer.input_size = 0;
        	//layer.output_size = 0;
        	return *this;
        }*/
        
        
        DeviceMatrix Layer::forward() {
        
        	if (this->input.totalSize() == 0)
        		throw std::runtime_error("Input matrix is empty");
        
        	
        	// W*x ++ biases;
        
        	DeviceMatrix output = DeviceMatrix::matMul(this->weights, this->input);
        	output = DeviceMatrix::broadcastAdd(output, this->biases);
        
        	return output;
        
        }
        
        DeviceMatrix Layer::backpropagate(DeviceMatrix nablaC) {  // backpropagation only, no training step
        	if (nablaC.rows() != this->outputSize())
        		throw std::runtime_error("Gradient output must have the same number of rows as the layer's output size");
        
        	return DeviceMatrix::matMul(DeviceMatrix::matTranspose(this->weights), nablaC);
        }
        
        DeviceMatrix Layer::updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate) {  // backpropagation + training step
        
        	if (gradient_output.rows() != this->outputSize())
        		throw std::runtime_error("Gradient output must have the same number of rows as the layer's output size");
        
        	// gradients calculation
        	DeviceMatrix gradient_weights = DeviceMatrix::matMul(gradient_output, DeviceMatrix::matTranspose(this->input));
        	DeviceMatrix gradient_biases = DeviceMatrix::matColReduce(gradient_output);
        	DeviceMatrix gradient_input = this->backpropagate(gradient_output);
        
        	// scaling gradients
        	gradient_weights = DeviceMatrix::matScale(gradient_weights, learning_rate);
        	gradient_biases = DeviceMatrix::matScale(gradient_biases, learning_rate);
        
        	// training step
        	weights = DeviceMatrix::matSub(weights, gradient_weights);
        	biases = DeviceMatrix::matSub(biases, gradient_biases);
        
        	return gradient_input;
        }
        
        
        void Layer::clean() noexcept {
        	this->weights.clean();
        	this->biases.clean();
        	this->input.clean();
        	//this->input_size = 0;
        	//this->output_size = 0;
        }
        

        Layer Layer::RandomLayer(size_t input_size, size_t output_size, std::pair<float, float> range) {
        
        	if (input_size == 0 || output_size == 0)
        		throw std::runtime_error("Input and output sizes must be greater than zero");
        
        	DeviceMatrix Weights = DeviceMatrix::Random(output_size, input_size, range);
        	DeviceMatrix Biases = DeviceMatrix::Random(output_size, 1, range);
        
        	return Layer(std::move(Weights), std::move(Biases));
        }

        void Layer::print(const char* header) const {
            
            Matrix weights_host = this->weights.downloadToHost();
            Matrix biases_host = this->biases.downloadToHost();
            
            printf("%s\n__________\n", header);
			printf("input size: %zu\n", this->inputSize());
			printf("output size: %zu\n", this->outputSize());
			printf("parameter count: %zu\n", this->parameterCount());
            printf("\n__________\n");
            
            printf("Weights:\n\n");

            weights_host.print(10, 10);
            

            printf("\n__________\nBiases:\n\n");
            biases_host.print(10, 10);
			printf("\n__________\n\n");
		}

        std::shared_ptr<LearningUnit> Layer::clone() const {
            return std::make_shared<Layer>(*this);
        }
        
	}; // namespace MLP      
};  // namespace dl