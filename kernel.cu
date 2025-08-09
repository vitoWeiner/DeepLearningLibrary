
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "./include/Matrix.hpp"
#include "./include/DeviceMatrix.cuh"
#include "./include/Models/MLP/Layer.cuh"




int main()
{

	MLP::Layer layer = MLP::Layer::RandomLayer(3, 2, { -1.0f, 1.0f });

	layer.setInput(DeviceMatrix({ 1.0f, 2.0f, 1.0f }, 3, 1));

	layer.forward().downloadToHost().print();

	

    return 0;
}

