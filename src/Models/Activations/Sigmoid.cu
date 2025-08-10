
#include "../../../include/Models/Activations/Sigmoid.cuh"

namespace dl {

	Sigmoid::Sigmoid() : input(), output() {}

	Sigmoid::Sigmoid(DeviceMatrix& input_matrix) : input(input_matrix) {}

	Sigmoid::Sigmoid(DeviceMatrix&& input_matrix) : input(std::move(input_matrix)) {}



};
