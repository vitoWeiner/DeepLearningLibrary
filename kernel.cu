
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


/*
interface:

copy ctor = deep copy
copy assignment = deep copy
clone = deep copy (return shared_ptr)
move ctor = ownership transfer
move assignment = ownership transfer
shared_ptr<LearningUnit> + shared_ptr<LearningUnit> = shared_ptr<Model> | Model.units = {shared_ptr, shared_ptr}
Model.bind = operator+

training data se moze djeliti (shared_ptr)
Model(vector<shared_ptr<LearningUnit>>) = radi shallow copy, samo pointere uzima na postojece learning unite, ovo omogucuje djeljenje parametara ali zahtjeva vecu odgovornost
svi ostalo konstruktori u pravilu rade deep copy

*/

#include <iostream>



void f() {

	{


		Model model({
			std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 4)),
			std::make_shared<ReLU>(),
			std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(4, 6)),
			std::make_shared<ReLU>(),
			std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(6, 4)),
			std::make_shared<Sigmoid>(),
			std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(4, 1)),
			std::make_shared<Sigmoid>()
			});


		std::shared_ptr<TrainingData> data = std::make_shared<TrainingData>();

		for (float i = 0; i < 0.5f; i += 0.1f) {
			for (float j = 0; j < 0.5f; j += 0.1f) {

				data->add({ i, j }, { i + j });
			}
		}

		model.setTrainingData(data);
		model.setCostFunction(std::make_shared<BCE>());


		model.trainMiniBatchSGD(2000, 28, 0.05);

		model.evaluate();


		std::cout << "first time : " << DeviceMatrix::instances << std::endl;



	}


}


int main()
{

	//f();
	
	
	for (int i = 1; i < 0; ++i) {


	DeviceMatrix m = DeviceMatrix::Random(100, i);

	DeviceMatrix a = DeviceMatrix::matColReduce(m);
	DeviceMatrix b = DeviceMatrix::matColReduceV2(m);

	if (a.downloadToHost() != b.downloadToHost()) {
		printf("everithings wrong");
	}

	printf("case %d is good\n", i);
	
	
}

	/*

	Model model({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 500)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(500, 300)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(300, 15)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(15, 1)),
		std::make_shared<Sigmoid>()
		});

	std::shared_ptr<TrainingData> data = std::make_shared<TrainingData>();


	for (float x = 0.0; x < 0.5; x += 0.01) {
		for (float y = 0.0; y < 0.5; y += 0.01) {
			data->add({ x, y }, { x + y });
		}
	}

	

	model.setTrainingData(data);
	model.setCostFunction(std::make_shared<MSE>());

	model.trainSingleBatchGD(10000, 0.2f);

	model.print();

	model.evaluate();*/



    return 0;
}

