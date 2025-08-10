#pragma once

#include "./LearningUnit.cuh"
#include <vector>
#include <memory> // for std::shared_ptr
#include <initializer_list>

namespace dl {
    
    class Model : public LearningUnit {

    protected:
		std::vector<std::unique_ptr<LearningUnit>> learning_units;
        // mora bit unique jer ako je shared_ptr onda bi znacilo da vise modela moze djeliti isti dio, sto nema smisla, svaki model je izgraden od unikatnih djelova

    public:

		Model() = default;

        Model(const Model& other);

        Model(Model&& other) noexcept;


        // ovo moram dalje
      /*  Model(std::initializer_list<std::unique_ptr<LearningUnit>> layers) {
            
            for (const auto& layer : layers) {
                if (!layer) {
                    throw std::runtime_error("Cannot bind a null LearningUnit.");
                }
                learning_units.push_back(layer);
			}
        }*/

        size_t outputSize() const override;

        size_t depth() const;

        void bind(const std::unique_ptr<LearningUnit>& unit);

        void bind(std::unique_ptr<LearningUnit>&& unit);

        DeviceMatrix forward() override;

        std::unique_ptr<LearningUnit> clone() const override;

        void clean() noexcept override;
        
        void print(const char* header = "\nModel analytics:") const;
        
        

	};  // class Model


    std::unique_ptr<Model> operator+(std::unique_ptr<LearningUnit>&& left, std::unique_ptr<LearningUnit>&& right);

    /*std::unique_ptr<Model> operator+(const std::unique_ptr<LearningUnit>& left, const std::unique_ptr<LearningUnit>& right);
	std::unique_ptr<Model> operator+(std::unique_ptr<LearningUnit>&& left, std::unique_ptr<LearningUnit>&& right);
	std::unique_ptr<Model> operator+(const std::unique_ptr<LearningUnit>& left, std::unique_ptr<LearningUnit>&& right);
	std::unique_ptr<Model> operator+(std::unique_ptr<LearningUnit>&& left, const std::unique_ptr<LearningUnit>& right);*/


};  // namespace dl 