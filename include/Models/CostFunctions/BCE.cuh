#pragma once

#include "./CostFunction.cuh"

namespace dl {
    class BCE : public CostFunction {
    public:
        DeviceMatrix compute(const DeviceMatrix& predictions, const DeviceMatrix& targets) override {
            return DeviceMatrix::BCE(predictions, targets);
        }
        DeviceMatrix gradient(const DeviceMatrix& predictions, const DeviceMatrix& targets) override {
            return DeviceMatrix::BCEGradient(predictions, targets);
        }
    }; // class BCE
}; // namespace dl



