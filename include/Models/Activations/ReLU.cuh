#pragma once

#include "../../DeviceMatrix.cuh"
#include "../LearningUnit.cuh"


namespace dl
{
	class ReLU : public LearningUnit {
	protected:
		DeviceMatrix input;
		DeviceMatrix output;
	public:
		ReLU();

		void setInput(const DeviceMatrix& input_matrix) override;
		void setInput(DeviceMatrix&& input_matrix) override;

		DeviceMatrix forward() override;
		DeviceMatrix backpropagate(DeviceMatrix gradient_output) override;
	
		DeviceMatrix  updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate = 0.01f) override;


		size_t parameterCount() const override { return 0; }
		bool has_variable_input() const override { return true; }
		bool backpropNeedsInput() const override { return false; }
		void print(const char* header = "\nReLU analytics:") const override {}; 
	};

}