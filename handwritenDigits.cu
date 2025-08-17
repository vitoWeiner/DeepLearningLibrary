#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <vector>
#include <cstdio>

#include "./include/Matrix.hpp"
#include "./include/DeviceMatrix.cuh"
#include "./include/Models/MLP/Layer.cuh"
#include "./include/Models/Model.cuh"
#include "./include/Models/Activations/Sigmoid.cuh"
#include "./include/Models/CostFunctions/MSE.cuh"
#include "./include/Models/TrainingData/MLP_TrainingData.cuh"
#include "./include/Models/Activations/ReLU.cuh"
#include "./include/Models/CostFunctions/BCE.cuh"

using namespace dl;

static std::shared_ptr<Model> mlp2;
static std::shared_ptr<TrainingData> data2;

extern "C" void initTrainingHWD() {

    mlp2 = std::make_shared<Model>(Model({
       std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(28 * 28, 16,  {-2, 2})),
       std::make_shared<Sigmoid>(),
       std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(16, 16, {-2, 2})),
         std::make_shared<Sigmoid>(),
       std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(16, 2,  {-2, 2})),
       std::make_shared<Sigmoid>()
        }));

    data2 = std::make_shared<TrainingData>();

    mlp2->setCostFunction(std::make_shared<MSE>());
    //mlp2->setTrainingData(data2);

}

extern "C" void initData() {
    mlp2->setTrainingData(data2);
}

extern "C" void addToData(float* p, int target) {
    Matrix m(p, 28 * 28, 1);
    Matrix s(2, 1);

    switch (target) {
    case 2: {
        s.setAt(1, 0, 0);
        s.setAt(0, 1, 0);
        break;
    }
    case 3: {
        s.setAt(0, 0, 0);
        s.setAt(1, 1, 0); 
        break;
    }
    default: {
        return;
    }

    }

    data2->add(m.toStdVector(), s.toStdVector());
}

extern "C" void train() {
    mlp2->trainSingleBatchGD(10000, 0.5);
    //mlp2->evaluate();
}

extern "C" void clearData() {
    data2->clear();
}

extern "C" void eval() {
    mlp2->evaluate();
}