#include <iostream>

#include "Matrix.hpp"
#include "DeviceMatrix.cuh"
#include "Models/MLP/Layer.cuh"
#include "Models/Model.cuh"
#include "Models/Activations/Sigmoid.cuh"
#include "Models/CostFunctions/MSE.cuh"
#include "Models/TrainingData/MLP_TrainingData.cuh"
#include "Models/CostFunctions/BCE.cuh"
#include "Models/Activations/ReLU.cuh"


using namespace dl;


int main_2() {

	Model model({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 2)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 1)),
		std::make_shared<Sigmoid>()
		});

	std::shared_ptr<TrainingData> data = std::make_shared<TrainingData>();

	for (float i = 0.0f; i < 0.5f; i += 0.1f) {
		for (float j = 0.0f; j < 0.5f; j += 0.1f) {
			data->add({ i, j }, { i + j });
		}
	}

	model.setTrainingData(data);
	model.setCostFunction(std::make_shared<MSE>());

	model.trainSingleBatchGD(4000, 0.5);

	model.print();
	model.evaluate();

	return 0;
}