#pragma once

#include <stdexcept> // for std::runtime_error
#include <memory>   // for std::unique_ptr

#include "../DeviceMatrix.cuh"


namespace dl {

	class LearningUnit {  // this is neutral unit, in MLP it will do f(input) = input, usage is to be base class for composite pattern in Model
		

/*
		
Grammar for LearningUnit:

LearningUnit ::= Model | Layer | ActivationFunction | LossFunction
Model ::= LearningUnit+


*/

	protected:
		DeviceMatrix input;

	    
	public:

		LearningUnit() = default;

		LearningUnit(const LearningUnit& other) : input(other.input) {}
		

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


			input = std::move(input_matrix);
			
		}

		virtual DeviceMatrix forward() {
			
			return this->input;
		}

		virtual DeviceMatrix backpropagate(const DeviceMatrix& gradient_output) {
			
			return gradient_output;
		}

		virtual  DeviceMatrix train(const DeviceMatrix& gradient_output, float learning_rate = 0.01f) {
			
			return gradient_output;
		}

		virtual size_t inputSize() const {
			return input.rows();
		}

		virtual size_t outputSize() const {
			return input.rows();
		}

		virtual std::unique_ptr<LearningUnit> clone() const {
			return std::make_unique<LearningUnit>(*this);
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