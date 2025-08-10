#include "../../include/Models/Model.cuh"
#include "../../include/Models/MLP/Layer.cuh"
#include <type_traits>

namespace dl {


    // Model::copy constructor
	Model::Model(const Model& other) {
		for (const auto& unit : other.learning_units) {
			learning_units.push_back(unit->clone());
		}
	}

    Model::Model(Model&& other) noexcept {
    
		learning_units = std::move(other.learning_units);

        other.clean();
    
    }

    size_t Model::inputSize() const {
                
        if (this->learning_units.empty()) {
            return LearningUnit::inputSize();
		}
        else {
            return learning_units.front()->inputSize();
		}
    }

    size_t Model::outputSize() const {
        
        if (this->learning_units.empty()) {
			return LearningUnit::outputSize();
		}
        else {
            return learning_units.back()->outputSize();
        }
	}



	void Model::bind(const std::unique_ptr<LearningUnit>& unit) {

		if (!unit) {
			throw std::runtime_error("Cannot bind a null LearningUnit.");
		}

        if (this->learning_units.size() > 0 && this->learning_units.back()->outputSize() != unit->inputSize()) {
            throw std::runtime_error("Input size of the new LearningUnit does not match the output size of the last LearningUnit in the model.");
        }


		learning_units.push_back(unit->clone());
	}


	void Model::bind(std::unique_ptr<LearningUnit>&& unit) {

		if (!unit) {
			throw std::runtime_error("Cannot bind a null LearningUnit.");
		}

        size_t f = this->learning_units.size();
        size_t s = (f == 0)? 0 : this->learning_units.back()->outputSize();
		size_t t = unit->inputSize();

        if (this->learning_units.size() > 0 && this->learning_units.back()->outputSize() != unit->inputSize()) {
            throw std::runtime_error("Input size of the new LearningUnit does not match the output size of the last LearningUnit in the model.");
		}

		learning_units.push_back(std::move(unit));

	}


    // FORWARD PASS

	DeviceMatrix Model::forward() {

       

		DeviceMatrix output = this->input;

		for (std::unique_ptr<LearningUnit>& unit : this->learning_units) {

			unit->setInput(std::move(output));
			output = unit->forward();
		}

		return output;
	}

    void Model::clean() noexcept {

        LearningUnit::clean();

        for (auto& unit : learning_units) {
            unit->clean();
        }

        learning_units.clear();
    }

    
    
    size_t Model::depth() const {

        // for each learning unit in learning_units, if learning unit is Model, add its depth to the total depth
        size_t total_depth = 0;

        for (const auto& unit : learning_units) {
            if (auto model_unit = dynamic_cast<Model*>(unit.get())) {
                total_depth += model_unit->depth();
            }
            else {
                total_depth++;
            }
        }

        return total_depth;
    }


    std::unique_ptr<LearningUnit> Model::clone() const {
		return std::make_unique<Model>(*this);  // this uvjek da lval ref, ovo poziva copy konstruktor za Model
    }


    void Model::print(const char* header) const {
        printf("\nModel analytics:\n");
        printf("\n__________\n");
        printf("Number of LearningUnits: %zu\n", learning_units.size());
        printf("Input size: %zu\n", inputSize());
        printf("Output size: %zu\n", outputSize());
        printf("Depth: %zu\n", depth());
        for (size_t i = 0; i < learning_units.size(); ++i) {
            printf("\nLearningUnit %zu:\n", i + 1);
            learning_units[i]->print();
		}

    }


	// OPERATOR OVERLOADS FOR MODEL COMPOSITION

    std::unique_ptr<Model> operator+(std::unique_ptr<MLP::Layer>&& left, std::unique_ptr<Model>&& right) {
        if (!left || !right) {
            throw std::runtime_error("Cannot add null LearningUnits.");
        }

        right->bind(std::move(left));

		return std::make_unique<Model>(std::move(*right));
    }

    std::unique_ptr<Model> operator+(std::unique_ptr<Model>&& left, std::unique_ptr<MLP::Layer>&& right) {
        if (!left || !right) {
            throw std::runtime_error("Cannot add null LearningUnits.");
        }
        left->bind(std::move(right));
    
		return std::make_unique<Model>(std::move(*left));
    }

    std::unique_ptr<Model> operator+(std::unique_ptr<LearningUnit>&& left, std::unique_ptr<LearningUnit>&& right) {
        if (!left || !right) {
            throw std::runtime_error("Cannot add null LearningUnits.");
        }

        Model model;

        model.bind(std::move(left));
        model.bind(std::move(right));

        return std::make_unique<Model>(std::move(model));
    }
   



};