#pragma once

#include "./LearningUnit.cuh"
#include <vector>
#include <memory> // for std::shared_ptr
#include <initializer_list>
#include "./MLP/Layer.cuh"
#include <stdexcept> // for std::runtime_error
#include <utility> // for std::move
#include "./CostFunctions/CostFunction.cuh"
#include "./TrainingData/MLP_TrainingData.cuh"

namespace dl {
    
    class Model : public LearningUnit {

    protected:
		std::vector<std::shared_ptr<LearningUnit>> learning_units;
        std::shared_ptr<CostFunction> cost_function;
        std::shared_ptr<TrainingData> training_data;
        // mora bit unique jer ako je shared_ptr onda bi znacilo da vise modela moze djeliti isti dio, sto nema smisla, svaki model je izgraden od unikatnih djelova

    public:

		Model() = default;

        Model(const Model& other); // = default;
		Model& operator=(const Model& other);

        Model(Model&& other) noexcept = default;
		Model& operator=(Model&& other) noexcept = default;

        Model(std::initializer_list<std::shared_ptr<LearningUnit>> layers);

        Model(const std::vector<std::shared_ptr<LearningUnit>>& layers);

   


        // ovo moram dalje
      /*  Model(std::initializer_list<std::shared_ptr<LearningUnit>> layers) {
            
            for (const auto& layer : layers) {
                if (!layer) {
                    throw std::runtime_error("Cannot bind a null LearningUnit.");
                }
                learning_units.push_back(layer);
			}
        }*/


        void setCostFunction(std::shared_ptr<CostFunction> cost_function);
        void setTrainingData(std::shared_ptr<TrainingData> training_data);  // zasad ne treba polimorfizam, pa je u redu ovako

        size_t depth() const;

        std::shared_ptr<Model> bind(std::shared_ptr<LearningUnit> unit);

        DeviceMatrix forward() override;

        DeviceMatrix updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate = 0.01f) override;

        DeviceMatrix backpropagate(DeviceMatrix gradient_output) override;

        void trainSingleBatchGD(size_t epochs, float learning_rate = 0.01f);

        void evaluate();

        std::shared_ptr<LearningUnit> clone() const override;

        void clean() noexcept override;
        
        void print(const char* header = "\nModel analytics:") const;

        size_t inputSize() const override;

        size_t outputSize() const override;

        size_t parameterCount() const override;

        friend std::shared_ptr<LearningUnit> operator+(const std::shared_ptr<LearningUnit> & model, const std::shared_ptr<LearningUnit>& unit);

	};  // class Model


	


};  // namespace dl 