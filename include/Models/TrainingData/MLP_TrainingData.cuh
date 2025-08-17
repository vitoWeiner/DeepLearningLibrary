#pragma once

#include "../../DeviceMatrix.cuh"

namespace dl {
    class TrainingData {

    protected:
        std::vector<std::vector<float>> inputs;

        std::vector<std::vector<float>> outputs;

    public:
        TrainingData();

        TrainingData(const TrainingData& other) = default;
        TrainingData& operator=(const TrainingData& other) = default;
        TrainingData(TrainingData&& other) noexcept = default;
        TrainingData& operator=(TrainingData&& other) noexcept = default;

        void add(const std::vector<float>& input, const std::vector<float>& output);

        size_t sampleCount();

        bool validate() const;
        void clear();

        std::pair<std::vector<DeviceMatrix>, std::vector<DeviceMatrix>> getMiniBatchesShuffled(size_t batch_size);
        std::pair<DeviceMatrix, DeviceMatrix> getSingleBatches();

        DeviceMatrix getOutputSamples();
        DeviceMatrix getInputSamples();

        ~TrainingData() = default;
    
    };


};
