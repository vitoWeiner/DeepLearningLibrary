#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <vector>
#include <cstdio>

#include "Matrix.hpp"
#include "DeviceMatrix.cuh"
#include "Models/MLP/Layer.cuh"
#include "Models/Model.cuh"
#include "Models/Activations/Sigmoid.cuh"
#include "Models/CostFunctions/MSE.cuh"
#include "Models/TrainingData/MLP_TrainingData.cuh"
#include "Models/Activations/ReLU.cuh"
#include "Models/CostFunctions/BCE.cuh"

using namespace dl;

static std::shared_ptr<Model> mlp2;
static std::shared_ptr<TrainingData> data2;

extern "C" void initTrainingHWD() {

    mlp2 = std::make_shared<Model>(Model({
       std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(28 * 28, 128,  {-3, 3})),
       std::make_shared<Sigmoid>(),
       std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(128, 64, {-3, 3})),
         std::make_shared<Sigmoid>(),
          std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(64, 10, {-3, 3})),
         std::make_shared<Sigmoid>(),

   
        }));

    data2 = std::make_shared<TrainingData>();

    mlp2->setCostFunction(std::make_shared<MSE>());
    //mlp2->setTrainingData(data2);

}

extern "C" void initData() {
    mlp2->setTrainingData(data2);
}

extern "C" void addToData(float* p, int target) {
    
    size_t outputsCount = 10;

    Matrix m(p, 28 * 28, 1);
    Matrix s(outputsCount, 1);

    if (target >= 0 && target < outputsCount)
        s.setAt(1, target, 0);

    data2->add(m.toStdVector(), s.toStdVector());
}

extern "C" void train() {
    mlp2->trainMiniBatchSGD(1000, 3000, 0.5);
    //mlp2->evaluate();
}

extern "C" void clearData() {
    data2->clear();
}

extern "C" void eval() {
    mlp2->evaluate();
}

extern "C" int predictDigit(float* p) {
    Matrix m(p, 28 * 28, 1);
    DeviceMatrix dm(m);

    mlp2->setInput(m);
    DeviceMatrix result = mlp2->forward();
    Matrix res = result.downloadToHost();
    

    int nClasses = res.rows(); 
    int bestClass = 0;
    float maxProb = res.getAt(0, 0);

    for (int i = 1; i < nClasses; i++) {
        float prob = res.getAt(i, 0);
        if (prob > maxProb) {
            maxProb = prob;
            bestClass = i;
        }
    }

    return bestClass;

}