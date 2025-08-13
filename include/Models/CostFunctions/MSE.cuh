
#pragma once

#include "./CostFunction.cuh"

namespace dl {
	class MSE : public CostFunction {

		DeviceMatrix compute(const DeviceMatrix& predictions, const DeviceMatrix& targets) override {

			return DeviceMatrix::MSE(predictions, targets);
		}

		DeviceMatrix gradient(const DeviceMatrix& predictions, const DeviceMatrix& targets) {
			return DeviceMatrix::MSEGradient(predictions, targets);
		}

	}; // class MSE
};  // namespace dl