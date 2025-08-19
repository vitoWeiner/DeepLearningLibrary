
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

int main_1() {

	Model AND_model({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 2)),   
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 1)),
		std::make_shared<Sigmoid>()
		});

	//AND_model.print();  // printing parameters

	std::shared_ptr<TrainingData> AND_data = std::make_shared<TrainingData>();

	AND_data->add({ 1, 1 }, { 1 });
	AND_data->add({ 0, 1 }, { 0 });
	AND_data->add({ 0, 1 }, { 0 });
	AND_data->add({ 0, 0 }, { 0 });

	AND_model.setTrainingData(AND_data);
	AND_model.setCostFunction(std::make_shared<MSE>());
	AND_model.trainSingleBatchGD(10000, 0.5);

	AND_model.evaluate();


	std::cin.get();


	Model XOR_model({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 2)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 1)),
		std::make_shared<Sigmoid>()
		});
	

	std::shared_ptr<TrainingData> XOR_data = std::make_shared<TrainingData>();
	

	XOR_data->add({ 0, 1 }, { 1 });
	XOR_data->add({ 1, 0 }, { 1 });
	XOR_data->add({ 1, 1 }, { 0 });
	XOR_data->add({ 0, 0 }, { 0 });

	XOR_model.setTrainingData(AND_data);
	XOR_model.setCostFunction(std::make_shared<MSE>());
	XOR_model.trainMiniBatchSGD(2000, 2, 0.5);


	XOR_model.evaluate();









	return 0;
}


