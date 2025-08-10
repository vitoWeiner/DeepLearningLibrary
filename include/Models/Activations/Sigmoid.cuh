#pragma once

#include "../../DeviceMatrix.cuh"

namespace dl {

    class Sigmoid {

    protected:
        DeviceMatrix input;
        DeviceMatrix output;
    
    public:
        Sigmoid();
        Sigmoid(DeviceMatrix& input_matrix);
        Sigmoid(DeviceMatrix&& input_matrix);
    
    
    };
};