
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



int main()
{
	

	/*Model model({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(5, 20)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(20, 20)),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(20, 70)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(70, 1))
		});


	model.print();
	*/

	



	Model xorModel({
	std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 2)), // bias obavezno
	std::make_shared<Sigmoid>(),
	std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 1)),
	std::make_shared<Sigmoid>() // za izlaz + BCE loss
		});



	std::shared_ptr<TrainingData> data = std::make_shared<TrainingData>();

	data->add({ 1, 0 }, { 1 });
	data->add({ 0, 1 }, { 1 });
	data->add({ 0, 0 }, { 0 });
	data->add({ 1, 1 }, { 0 });

	xorModel.setTrainingData(data);
	xorModel.setCostFunction(std::make_shared<BCE>());

//	model.print();

	xorModel.trainSingleBatchGD(5000, 0.05f);

	xorModel.evaluate();

	

	

	//model->setInput(DeviceMatrix::Random(5, 1));

	//model->forward();



	// make_model(LearningUnits);
	// Model += {Unit, Unit, Unit}
	// combine_models(Model&& x, Model&& y);


	
		                            



	/*

	MLP::Layer layer = MLP::Layer::RandomLayer(20, 20);
	

	layer.setInput(DeviceMatrix::Random(20, 1, { -10.0f, 1.0f }));

	MLP::Layer layer2(DeviceMatrix::Random(20, 20), DeviceMatrix::Random(20, 1));
	
	layer2.setInput(layer.forward());

	

	std::shared_ptr<Model> model;
	
	std::shared_ptr<Model> model2 =  model->clone() + model->clone();


	*/
	

	//printf("%zu", model2->depth());

    return 0;
}

