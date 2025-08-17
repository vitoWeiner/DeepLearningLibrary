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

using namespace dl;

static std::shared_ptr<Model> mlp;
static DeviceMatrix g_output;
static std::shared_ptr<TrainingData> data;
static bool g_swapXY = false;

extern "C" void setOutputSwap(bool swap) { g_swapXY = swap; }

extern "C" void initTraining(float* img1_data, float* img2_data, int width, int height) {
    // model: input 3 -> hidden 64 -> 64 -> 28 -> 1 with sigmoid
    mlp = std::make_shared<Model>(Model({
        std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(3, 64,  {-8, 8})),
        std::make_shared<Sigmoid>(),
        std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(64, 64, {-8, 8})),
        std::make_shared<Sigmoid>(),
        std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(64, 28, {-8, 8})),
        std::make_shared<Sigmoid>(),
        std::make_shared<MLP::Layer>(MLP::Layer::RandomLayer(28, 1,  {-8, 8})),
        std::make_shared<Sigmoid>()
        }));

    Matrix out1(img1_data, height, width);
    Matrix out2(img2_data, height, width);

    data = std::make_shared<TrainingData>();
    //data->reserve(width * height * 2);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float nx = (static_cast<float>(j) / (width - 1)) * 2.0f - 1.0f;
            float ny = (static_cast<float>(i) / (height - 1)) * 2.0f - 1.0f;

            data->add({ -1.0f, nx, ny }, { out1.getAt(i, j) });
            data->add({ -1.0f + 0.2, nx, ny }, { out2.getAt(i, j) });
        }
    }

    mlp->setTrainingData(data);
    mlp->setCostFunction(std::make_shared<MSE>());
}

extern "C" void trainStep() {
    if (!mlp) return;
    mlp->trainSingleBatchGD(10, 2);
}

extern "C" float getCost() {
    if (!mlp) return 0.0f;
    return mlp->computeCost();
}
extern "C" void getCurrentOutput(float* out_buffer, float flag, int width, int height) {
    if (!mlp) return;

    const int N = width * height;
    std::vector<float> input_vec(3 * N);

    int i = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
            float nx = (static_cast<float>(x) / (width - 1)) * 2.0f - 1.0f;
            float ny = (static_cast<float>(y) / (height - 1)) * 2.0f - 1.0f;

            // row-major 3xN: first all flags, then all nx, then all ny
            input_vec[i] = flag;
            input_vec[N + i] = nx;
            input_vec[2 * N + i] = ny;
        }
    }

    DeviceMatrix d_in(Matrix(input_vec.data(), 3, N));
    mlp->setInput(std::move(d_in));
    g_output = mlp->forward();

    std::vector<float> tmp(N);
    g_output.downloadToHost(tmp.data());

    if (!g_swapXY) {
        for (int k = 0; k < N; ++k) out_buffer[k] = tmp[k];
    }
    else {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int dst = y * width + x;
                int src = x * height + y;
                out_buffer[dst] = tmp[src];
            }
        }
    }
}
