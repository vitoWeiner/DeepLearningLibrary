#include "../../include/Models/Model.cuh"
#include "../../include/Models/MLP/Layer.cuh"
#include <type_traits>
#include <initializer_list>
#include <vector>
#include <stdexcept>

namespace dl {


    // Model::copy constructor
	Model::Model(const Model& other) {
		for (const auto& unit : other.learning_units) {
			learning_units.push_back(unit->clone());
		}
	}

    Model& Model::operator=(const Model& other) {
        
        if (this == &other) {
            return *this; // self-assignment check
        }
        
        this->clean();

       
        for (const auto& unit : other.learning_units) {
            learning_units.push_back(unit->clone());
        }
		return *this;


    }


/*   Model::Model(Model&& other) noexcept {
    
		learning_units = std::move(other.learning_units);

        other.clean();
    
    } */

    Model::Model(std::initializer_list<std::shared_ptr<LearningUnit>> layers) : Model(std::vector<std::shared_ptr<LearningUnit>>(layers)) {}


    Model::Model(const std::vector<std::shared_ptr<LearningUnit>>& layers) {  // warning: shallow copy, not move


        if (layers.empty())
            return;

        bool previous_has_variable_input = true;
        size_t previous_layer_output_size = layers.front()->inputSize();

        for (const auto& layer : layers) {
            if (!layer) {
                throw std::runtime_error("Cannot bind a null LearningUnit.");
            }

            if (!layer->has_variable_input()) {

                if (layer->inputSize() != previous_layer_output_size && !previous_has_variable_input) {
                    throw std::runtime_error("Input size of the new LearningUnit does not match the output size of the last LearningUnit in the model.");
			    }

                previous_has_variable_input = false;
                
			    previous_layer_output_size = layer->outputSize();
            }

            learning_units.push_back(layer);
        }

        if (learning_units.empty()) {
            throw std::runtime_error("Model cannot be initialized with an empty list of LearningUnits.");
        }
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

    size_t Model::parameterCount() const {

        size_t total_parameters = 0;
        for (const auto& unit : learning_units) {
            total_parameters += unit->parameterCount();
        }
		return total_parameters;
    }

    



	std::shared_ptr<Model> Model::bind(std::shared_ptr<LearningUnit> unit) {

		if (!unit) {
			throw std::runtime_error("Cannot bind a null LearningUnit.");
		}

        if (this->learning_units.size() > 0 && this->learning_units.back()->outputSize() != unit->inputSize()) {
            throw std::runtime_error("Input size of the new LearningUnit does not match the output size of the last LearningUnit in the model.");
        }

        std::shared_ptr<Model> model = std::make_shared<Model>();

        model->learning_units.push_back(std::make_shared<Model>(*this));
        model->learning_units.push_back(std::move(unit));

        return model;
	}


    // FORWARD PASS

	DeviceMatrix Model::forward() {

       

		DeviceMatrix output = this->input;

		for (std::shared_ptr<LearningUnit>& unit : this->learning_units) {

			unit->setInput(std::move(output));
			output = unit->forward();
		}

		return output;
	}

    DeviceMatrix Model::updateParamsAndBackpropagate(DeviceMatrix gradient_output, float learning_rate) {


		int n = this->learning_units.size();
        
        for (int layer = n - 1; layer >= 0; --layer) {

            gradient_output = this->learning_units[layer]->updateParamsAndBackpropagate(std::move(gradient_output), learning_rate);

        }

        return gradient_output;
    }

    DeviceMatrix Model::backpropagate(DeviceMatrix gradient_output) {
        if (this->learning_units.size() == 0) {
            return gradient_output;
        }

        DeviceMatrix output = std::move(gradient_output);

        for (int layer = this->learning_units.size() - 1; layer >= 0; --layer) {
            output = this->learning_units[layer]->backpropagate(std::move(output));
		}

        return output;
    }

    void Model::setCostFunction(std::shared_ptr<CostFunction> cost_function) {
		this->cost_function = std::move(cost_function);
    }

    void Model::setTrainingData(std::shared_ptr<TrainingData> training_data) {
        this->training_data = std::move(training_data);
        
    }

    void Model::trainSingleBatchGD(size_t epochs, float learning_rate) {

        if (this->cost_function == nullptr) {
            throw std::runtime_error("Cost function is not set. Please set a cost function before training.");
        }

        if (this->learning_units.empty()) {
            return;
        }

        if (this->training_data == nullptr) {
            throw std::runtime_error("Training data is not set. Please set training data before training.");
        }

		std::pair<DeviceMatrix, DeviceMatrix> batch = this->training_data->getSingleBatches();

	
		DeviceMatrix target_matrix = std::move(batch.second);

		this->setInput(std::move(batch.first));

        for (size_t epoch = 0; epoch < epochs; ++epoch) {

            DeviceMatrix output = this->forward();


            if (epoch % 10 == 0)
			    this->cost_function->compute(output, target_matrix).downloadToHost().print();

			DeviceMatrix gradient_output = this->cost_function->gradient(output, target_matrix);

            gradient_output = this->updateParamsAndBackpropagate(std::move(gradient_output), learning_rate);
        
        }



    }

    void Model::evaluate()  {

        if (this->cost_function == nullptr) {
            throw std::runtime_error("Cost function is not set. Please set a cost function before evaluating.");
        }
        if (this->learning_units.empty()) {
            return;
        }
        if (this->training_data == nullptr) {
            throw std::runtime_error("Training data is not set. Please set training data before evaluating.");
		}

        std::pair<DeviceMatrix, DeviceMatrix> batch = this->training_data->getSingleBatches();

		DeviceMatrix input_matrix = std::move(batch.first);
       
        DeviceMatrix target_matrix = std::move(batch.second);
        
        this->setInput(input_matrix);
        
        DeviceMatrix output = this->forward();
        DeviceMatrix cost = this->cost_function->compute(output, target_matrix);

		

		Matrix input_host = input_matrix.downloadToHost();
		Matrix output_host = output.downloadToHost();

        if (input_host.cols() != output_host.cols()) {
			throw std::runtime_error("Input and output matrices must have the same number of samples. Something is wrong!!!");
        }


        printf("\nEvaluation results:\n");

        printf("Input:\n");


        for (size_t sample = 0; sample < input_host.cols(); ++sample) {
            
            printf("\n__________\n");
            printf("\nSample %zu:\n", sample + 1);
            printf("\ninput:\n");
            for (size_t row = 0; row < input_host.rows(); ++row) {
				printf("%f ", input_host.getAt(row, sample));
            }
			printf("\noutput:\n");
            for (size_t row = 0; row < output_host.rows(); ++row) {
				printf("%f ", output_host.getAt(row, sample));
            }
            printf("\n__________\n");
						
        }

		printf("\nCost:\n");
		cost.downloadToHost().print();


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


    std::shared_ptr<LearningUnit> Model::clone() const {
		return std::make_shared<Model>(*this);  // this uvjek da lval ref, ovo poziva copy konstruktor za Model
    }

    


    void Model::print(const char* header) const {
        printf("\nModel analytics:\n");
        printf("\n__________\n");
        printf("Number of LearningUnits: %zu\n", learning_units.size());
        printf("Input size: %zu\n", inputSize());
        printf("Output size: %zu\n", outputSize());
        printf("Depth: %zu\n", depth());
		printf("Parameter count: %zu\n", parameterCount());
		printf("__________\n");

        for (size_t i = 0; i < learning_units.size(); ++i) {
            printf("\nLearningUnit %zu:\n", i + 1);
            learning_units[i]->print();
		}

    }


	// OPERATOR OVERLOADS FOR MODEL COMPOSITION

    std::shared_ptr<LearningUnit> operator+(const std::shared_ptr<LearningUnit>& left, const std::shared_ptr<LearningUnit>& right) {
        
		std::shared_ptr<Model> model = std::make_shared<Model>();

		model->learning_units.push_back(left);
        model->learning_units.push_back(right);

        return model;
    }



};