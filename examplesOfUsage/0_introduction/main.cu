/// EXAMPLE OF USAGE DEEP LEARNING LIB

#include <iostream>

#include "Matrix.hpp"   // this class is for doing matrix operations on CPU
#include "DeviceMatrix.cuh"   // this class is for doing matrix operation od CUDA GPU
#include "Models/MLP/Layer.cuh"   // class that represent perceptron layer : has reference on DeviceMatrix weights and biases
#include "Models/Model.cuh"    // logic for training
#include "Models/Activations/Sigmoid.cuh"  
#include "Models/CostFunctions/MSE.cuh"
#include "Models/TrainingData/MLP_TrainingData.cuh"
#include "Models/CostFunctions/BCE.cuh"
#include "Models/Activations/ReLU.cuh"



using namespace dl;

/// Introduction, how to use library

int main() {


/******************************************************************************

								examples of Matrix usage

*******************************************************************************/

	// Matrix is object on host memory, constructor is alocating heap memory for matrix, all operations are on CPU for this object

	Matrix noMat; // object matrix with 0 rows and 0 cols (no memory alocated)

	Matrix k(2, 2); // constructing matrix k = {{0, 0}, {0,0}}; (2 rows, 2 cols)

	Matrix x = Matrix::Random(2, 2, { -5, 5 });  // random 2x2 matrix with values from -5 to 5 for each element
	Matrix y = Matrix::Identity(2);  // Identity matrix 2x2

	// custom initialization:
	Matrix z({ 2, 0, 0, 2 }, 2, 2);

	// deep copy
	k = x;

	// moving ownership
	x = std::move(k);

	// deep equality
	if (z == y)
		printf("z and y are same");

	z.clean(); // dealocating resorces, setting size to zero, rows to zero, cols to zero


	// mat operation:


	y = y * 2; // scalar matrix multiplication on CPU
	
	Matrix sum = x + y;  // sum on CPU
	Matrix product = x * y;  // matmul on CPU
	product.print();  // printing on standard character output

	Matrix t = sum.transpose(); // transposing

	Matrix i = Matrix::Identity(4);
	Matrix i2 = Matrix::Identity(4);

	Matrix c_rows = Matrix::matConcatRows(i, i2);  // rows concatenation

	c_rows.print();

	Matrix c_cols = Matrix::matConcatCols(i, i2);  // cols concatenation

	

/*

for all operations check the header file Matrix.hpp in include directory

*/



/******************************************************************************

								examples of DeviceMatrix usage

*******************************************************************************/

	// constructing matrix on cpu for later:
	x = Matrix::Random(2, 2, { -5, 5 });
	y = Matrix::Identity(2);




   // DeviceMatrix is object on CUDA GPU, all operations are on GPU for this object, constructor is alocating heap memory on CUDA VRAM

	DeviceMatrix x_d = DeviceMatrix(x);  // constructing object DeviceMatrix on gpu based on CPU matrix, this is pure construction, there is no side effects onto cpu matrix (matrix is just deep copied on VRAM)
	DeviceMatrix y_d = DeviceMatrix(y);


	DeviceMatrix a_d = x_d; // deep copy
	 x_d = std::move(a_d); // ownership transfer


	DeviceMatrix product_d = DeviceMatrix::matMul(x_d, y_d);  // matmul on CUDA gpu, the CUDA kernel is called here

	Matrix product_h = product_d.downloadToHost(); // downloading matrix from device to host

	product_d.downloadToHost().print(); // its not posible to print matrix directly on device, need to move it to host

	// all methods type: DeviceMatrix::mat... (DeviceMatrix, DeviceMatrix); are calling CUDA kernel on matrices



	// DeviceMatrix can be constructed without Matrix object
	DeviceMatrix exmpl({ 1, 2, 3, 4 }, 2, 2);
	DeviceMatrix exmpl2 = DeviceMatrix::Random(2, 2, { -1.0f, 1.0f });  // 2 rows, 2 cols, random values from -1 to 1
	
	DeviceMatrix t_d = DeviceMatrix::matTranspose(exmpl2);  // kernel for transposing

	/**********************
	
	all method signatures are in the header file DeviceMatrix.cuh, all kernel definitions are in DeviceMatrixOps.cu
	
	***********************/
	


/******************************************************************************

								examples of Layer usage

*******************************************************************************/


	


	// Layer is object that represent one linear layer of MLP, with weights and biases, the constructor is constructing DeviceMatrix weights and biases, on CUDA VRAM heap

	MLP::Layer layer = MLP::Layer::RandomLayer(2, 2, { -1.0f, 1.0f });  // initializing layer with 2 inputs and 10 outputs, with random values from -1 to 1 
    
	layer.print();  // this can be costly because it downloads weights and biases to host side

	// Multiple layer can be chained in model using model constructor 



	DeviceMatrix batch = DeviceMatrix::Random(2, 20);
	DeviceMatrix targets = DeviceMatrix::Random(2, 20);

	layer.setInput(batch);  // setting input for forwarding

	/// CONVENTION IS: ROWS OF INPUT-MATRIX ARE DIMENSIONS of each input vector sample, COLS OF INPUT-MATRIX ARE vector SAMPLES

	// WITH THIS CONVENTION: IF THERE IS ONLY ONE DATA SAMPLE TO FEED FORWARD, IT WILL BE IN SHAPE  : DeviceMatrix sample(rows = VectorDim, cols = 1);

	// example of feed forward and backpropagating interface

	DeviceMatrix out = layer.forward();  // out = W * inputBatch ++ biases where ++ is broadcast sum over samples in batch

	DeviceMatrix simpleCostGradient = DeviceMatrix::matSub(targets, out); // calling kernel for targets - batch;


	DeviceMatrix gradientLayer = layer.updateParamsAndBackpropagate(simpleCostGradient);  

	/*
	method Layer.updateParamsAndBackpropagate : 
	    input = gradient of next layer to cost function
		output = gradient of current layer to cost function
		effect = updating weights and biases
	
	method Layer.backpropagate :
	    input = gradient of next layer to cost function
		output = gradient of current layer to cost function
		effect = there is no effect of this method

	*/

	// Layer inherit class LearningUnit

/******************************************************************************

								examples of Activation usage

*******************************************************************************/


	// Sigmoid object present sigmoid function and sigmoid gradient
	// s.forward = sigmoid(x)
	// s.backpropagate = sigmoidGradient(costGradient);

	Sigmoid s;

	s.setInput(DeviceMatrix::Random(2, 2));

	s.forward().downloadToHost().print();

	s.backpropagate(DeviceMatrix::Random(2, 2)).downloadToHost().print();  // calculating sigmoid gradient to the cost function

	// s.updateParamsAndBackpropagate() just calls backpropagate because there is no parameters to update


	/****
	
	
	Sigmoid has updateParamsAndBackpropagate method because it inherited it from the LearningUnit class. 
	The LearningUnit class represents an abstract class that is inherited by all 
	classes whose objects should be able to be 
	chained in MLP model (Layer, Sigmoid, etc...),
	the idea of library is that all learning units know how to do feed forward and backpropagating only themselves (Layer knows how to feed forward itself, how to update itself etc.., Sigmoid knows how to feed forward itself etc...),
	so when they are chained in MLP, output from one unit can be forward in next unit, and gradient of next unit can be backpropagate to current unit

	in this way its easier to add more learning units and compose new networks with new posibilities.
	All methods for learning units can be found in LearningUnit.cuh header file

	****/

	

/******************************************************************************

								examples of Model usage

*******************************************************************************/




// model is class thet accept pointers to learning units , examples of learning units are : Layer, Sigmoid, ReLU

	Model model({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(1, 1, {-10, 10}))   // model can be made from one or more LearningUnit objects (Layer, Sigmoid, ReLU,...)
		});  

	model.setInput(DeviceMatrix::Random(1, 5));  // setting up input with 1 dimension and 5 data samples

	model.forward().downloadToHost().print();


	// if outpout of layer not match input of next layer it will throw exception

	try {
		Model m({
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(2, 2)),
		std::make_shared<Sigmoid>(),
		std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(10, 1)), // 10 not match 2
		std::make_shared<Sigmoid>()
			});
	}
	catch (std::runtime_error x) {

		std::cout << x.what();
	
	}

	/// SIMPLE EXAMPLE OF TRAINING

	// setting upt training data:

	std::shared_ptr<TrainingData> data = std::make_shared<TrainingData>();

	data->add({ 0 }, { 0 });   // matching inputs to outputs for input 0, output is 0
	data->add({ 1 }, { 5 });  // for input 1 output is 5
	data->add({ 2 }, { 10 });  // for input 2 output is 10
	data->add({ 3 }, { 15 });  // for input 3 output is 15
	data->add({ 4 }, { 20 });


	model.setTrainingData(data);
	model.setCostFunction(std::make_shared<MSE>());

	model.print();

	model.trainSingleBatchGD(2000, 0.01);

	std::shared_ptr<TrainingData> test = std::make_shared<TrainingData>();

	model.print();



	test->add({ 0 }, { 0 });
	test->add({ 4 }, { 20 });
	test->add({ 5 }, { 25 });
	test->add({ 10 }, { 50 });
	test->add({ 100 }, { 500 });

	model.evaluate(test);

	// Model is also Learning Unit, can be chained inside other model in tree like structure
	

	/******************
	
	WARNINGS

	!!! model is using shared_ptr on layers and activations and cost function and training data, no deep copies.	
	****************/

	return 0;
}