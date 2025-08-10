
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "./include/Matrix.hpp"
#include "./include/DeviceMatrix.cuh"
#include "./include/Models/MLP/Layer.cuh"
#include "./include/Models/Model.cuh"

using namespace dl;


class Base {
public:
	int x;
};

class Derived : public Base { public: int y;  };

int main()
{




	MLP::Layer layer = MLP::Layer::RandomLayer(20, 20);
	

	layer.setInput(DeviceMatrix::Random(20, 1, { -10.0f, 1.0f }));

	MLP::Layer layer2(DeviceMatrix::Random(20, 20), DeviceMatrix::Random(20, 1));
	
	layer2.setInput(layer.forward());

	

	std::unique_ptr<Model> model = std::make_unique<MLP::Layer>(layer) 
		                         + std::make_unique<MLP::Layer>(layer) 
		                         + std::make_unique<MLP::Layer>(MLP::Layer::RandomLayer(20, 5));
	
	std::unique_ptr<Model> model2 = model->clone() + std::make_unique<MLP::Layer>(MLP::Layer::RandomLayer(5, 1));


	model->setInput(DeviceMatrix::Random(20, 1, { -10.0f, 1.0f }));

	printf("%zu", model->depth());

    return 0;
}

