#pragma once

#include <stdexcept> // for std::runtime_error
#include <memory>   // for std::shared_ptr

#include "../DeviceMatrix.cuh"


namespace dl {

	class LearningUnit {  // this is neutral unit, in MLP it will do f(input) = input, usage is to be base class for composite pattern in Model
		

/*
		
Composition tree for LearningUnit:

LearningUnit ::= Model | Layer | ActivationFunction | LossFunction
Model ::= LearningUnit*


*/

	protected:
		DeviceMatrix input;

	    
	public:

		LearningUnit() = default;

		LearningUnit(const LearningUnit& other) = default;
		LearningUnit& operator=(const LearningUnit& other) = default;
		LearningUnit(LearningUnit&& other) noexcept = default;
		LearningUnit& operator=(LearningUnit&& other) noexcept = default;


		

		virtual void setInput(const DeviceMatrix& input_matrix) {
			
			if (input_matrix.totalSize() == 0) {
				throw std::runtime_error("Input matrix cannot be empty.");
			}

			input = input_matrix;

		}

		virtual void setInput(DeviceMatrix&& input_matrix) {

			if (input_matrix.totalSize() == 0) {
				throw std::runtime_error("Input matrix cannot be empty.");
			}


			this->input = std::move(input_matrix);
			
		}

		virtual DeviceMatrix forward() {
			
			return this->input;
		}

		virtual DeviceMatrix backpropagate(DeviceMatrix gradient_output) {
			
			return gradient_output;
		}

		virtual  DeviceMatrix updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate = 0.01f) {
			
			return gradient_output;
		}

		virtual size_t inputSize() const {
			return input.rows();
		}

		virtual size_t outputSize() const {
			return input.rows();
		}

		virtual size_t parameterCount() const {
			return 0; 
		}

		virtual bool has_variable_input() const {  // does it works for any input or just for static size inputs? for example layers must have defined input size and output size, sigmoid not
			return true; 
		}

		virtual bool backpropNeedsInput() const { // can we do backpropagation on this unit, only if input is set (not a null-pointer)?    
			return true;
		}

		virtual std::shared_ptr<LearningUnit> clone() const {
			return std::make_shared<LearningUnit>(*this);
		}

		

		virtual void clean() noexcept {
			this->input.clean();
		}

		virtual void print(const char* header = "\nLearningUnit analytics:") const {
			printf("%s\n", header);
			printf("Input size: %zu\n", inputSize());
			printf("Output size: %zu\n", outputSize());
		}

		virtual ~LearningUnit() = default;
	
	}; // class LearningUnit


	

}; // namespace dl