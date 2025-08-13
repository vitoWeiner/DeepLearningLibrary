#pragma once

#include "../../DeviceMatrix.cuh"

namespace dl {


class CostFunction {

public:
	virtual DeviceMatrix compute(const DeviceMatrix& predictions, const DeviceMatrix& targets) = 0;
	virtual DeviceMatrix gradient(const DeviceMatrix& predictions, const DeviceMatrix& targets) = 0;
	virtual ~CostFunction() = default;
    
    };  // class CostFunction 
};