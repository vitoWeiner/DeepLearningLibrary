#include <utility> // for std::move


#include "../../../include/Models/Activations/ReLU.cuh"
#include "../../../include/DeviceMatrix.cuh"

namespace dl {

	ReLU::ReLU() {}


	void ReLU::setInput(const DeviceMatrix& input_matrix) {
		this->input = input_matrix;
	}

	void ReLU::setInput(DeviceMatrix&& input_matrix) {
		this->input = std::move(input_matrix);
	}


	DeviceMatrix ReLU::forward() {

		if (this->input.totalSize() == 0) {
			throw std::runtime_error("Input matrix is empty");
		}

		this->output = DeviceMatrix::ReLU(this->input);

		return this->output;
	}



	DeviceMatrix ReLU::backpropagate(DeviceMatrix gradient_output) {

		if (this->output.totalSize() == 0) {
			this->output = this->forward();
		}

		if (gradient_output.totalSize() == 0) {
			throw std::runtime_error("Gradient output matrix is empty");
		}

		DeviceMatrix output = DeviceMatrix::matElementWiseMul(DeviceMatrix::ReLUGradient(this->output), gradient_output);

		return output;
	}

	DeviceMatrix  ReLU::updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate) {
		return this->backpropagate(std::move(gradient_output));
	};




};
