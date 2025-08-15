#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "./include/Matrix.hpp"
#include "./include/DeviceMatrix.cuh"
#include "./include/Models/MLP/Layer.cuh"
#include "./include/Models/Model.cuh"
#include "./include/Models/Activations/Sigmoid.cuh"
#include "./include/Models/CostFunctions/MSE.cuh"
#include "./include/Models/TrainingData/MLP_TrainingData.cuh"
#include "./include/Models/CostFunctions/BCE.cuh"
#include "./include/Models/Activations/ReLU.cuh"

using namespace dl;


static std::shared_ptr<Model> mlp;
static DeviceMatrix g_input, g_target, g_output;
static std::shared_ptr<TrainingData> data;

extern "C" void initTraining(float* target_data, int width, int height) {
	mlp = std::make_shared<Model>(Model({
	std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 9, {-1, 1})),
	std::make_shared<Sigmoid>(),
	std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(9, 9, {-1, 1})),
	std::make_shared<Sigmoid>(),
	std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(9, 1, {-1, 1})),
	std::make_shared<Sigmoid>()
		}));



	Matrix out(target_data, height, width);

	/*bool x = out.check([](float x)->bool {
		return (x <= 1) && (x >= 0);
		});*/

	data = std::make_shared<TrainingData>();

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			data->add(
				{ static_cast<float>(i) / 27, static_cast<float>(j) / 27 },
				{ out.getAt(i, j) }
			);
		}
	}

	/*	bool y = data->getOutputSamples().downloadToHost().check([](float x)->bool {
			return (x <= 1) && (x >= 0);
			});*/

	mlp->setTrainingData(data);
	mlp->setCostFunction(std::make_shared<MSE>());

}

extern "C" void trainStep() {
	mlp->trainMiniBatchSGD(10, 24, 0.05F);
}

extern "C" float getCost() {
	return mlp->computeCost();
}

extern "C" void getCurrentOutput(float* out_buffer) {
	if (!mlp) return;

	g_output = mlp->forward(); // napravimo forward pass
	g_output.downloadToHost(out_buffer); // prebacimo u CPU buffer
}
