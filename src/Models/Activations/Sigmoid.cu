
#include "../../../include/Models/Activations/Sigmoid.cuh"

namespace dl {

	//Sigmoid::Sigmoid() : input(), output() {}

	Sigmoid::Sigmoid() {}

	DeviceMatrix Sigmoid::forward() {

		if (input.totalSize() == 0) {
			return input;
		}

		output = DeviceMatrix::Sigmoid(input);

		return output;
	}

	DeviceMatrix Sigmoid::backpropagate(DeviceMatrix gradient_output) {
		if (output.totalSize() == 0 || gradient_output.totalSize() == 0) {
			return DeviceMatrix();
		}
		
		DeviceMatrix sigmoid_gradient = DeviceMatrix::SigmoidGradient(output);
		DeviceMatrix grad_input = DeviceMatrix::matElementWiseMul(gradient_output, sigmoid_gradient);

		return grad_input;
	}


	DeviceMatrix Sigmoid::updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate) {
		if (output.totalSize() == 0 || gradient_output.totalSize() == 0) {
			return DeviceMatrix();
		}
	    
		return this->backpropagate(gradient_output);
		
	}


	void Sigmoid::print(const char* header) const {

		printf("%s\n", header);
		printf("\n________\n");
		printf("Sigmoid activation function\n");
		printf("Input size: any\n");
		printf("Output size: any\n");
		printf("Parameter count: no parameter\n");	
		printf("\n________\n");


	}
	

	//Sigmoid::Sigmoid(DeviceMatrix& input_matrix) : input(input_matrix) {}

	//Sigmoid::Sigmoid(DeviceMatrix&& input_matrix) : input(std::move(input_matrix)) {}



};
