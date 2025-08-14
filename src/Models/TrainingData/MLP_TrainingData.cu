#include "../../../include/Models/TrainingData/MLP_TrainingData.cuh"
#include <stdexcept>
#include <algorithm>
#include <random>
#include <utility>

namespace dl {
	TrainingData::TrainingData() {}

	void TrainingData::add(const std::vector<float>& input, const std::vector<float>& output) {

		if (input.empty() || output.empty())
			throw std::runtime_error("input is empty or output is empty");

		if (!inputs.empty()) {
			if (inputs.back().size() != input.size() || outputs.back().size() != output.size())
				throw std::runtime_error("inputs are not same size with other inputs or outputs are not the same size with other outputs");
		}

		inputs.push_back(input);
		outputs.push_back(output);
	}

	size_t TrainingData::sampleCount() {
		return inputs.size();
	}


	bool TrainingData::validate() const {

		if (inputs.size() != outputs.size()) {
			return false;
		}

		if (inputs.size() == 0)
			return true;

		size_t in_size = inputs.at(0).size();
		size_t out_size = outputs.at(0).size();

		for (const auto& in : inputs) {
			if (in.size() != in_size)
				return false;
		}

		for (const auto& out : outputs) {
			if (out.size() != out_size) {
				return false;
			}
		}

		return true;
	}

	DeviceMatrix TrainingData::getInputSamples() {
		if (!this->validate()) {
			throw std::runtime_error("from TrainingData::getSingleBatches -> problem : trainingData not valid");
		}


		if (inputs.empty()) {
			throw std::runtime_error("TrainingData::getSingleBatches called on empty dataset.");
		}

		Matrix input_batch(inputs[0].size(), inputs.size());

		for (size_t col = 0; col < inputs.size(); ++col) {
			for (size_t row = 0; row < inputs[col].size(); ++row) {

				input_batch.setAt(inputs[col][row], row, col);

			}
		}

		return DeviceMatrix(input_batch);

	}

	DeviceMatrix TrainingData::getOutputSamples() {

		if (!this->validate()) {
			throw std::runtime_error("from TrainingData::getSingleBatches -> problem : trainingData not valid");
		}


		if (inputs.empty()) {
			throw std::runtime_error("TrainingData::getSingleBatches called on empty dataset.");
		}

		Matrix output_batch(outputs[0].size(), outputs.size());

		for (size_t col = 0; col < inputs.size(); ++col) {
			
			for (size_t row = 0; row < outputs[col].size(); ++row) {

				output_batch.setAt(outputs[col][row], row, col);
			}
		}

		return DeviceMatrix(output_batch);

	}

	std::pair<DeviceMatrix, DeviceMatrix> TrainingData::getSingleBatches() {

		if (!this->validate()) {
			throw std::runtime_error("from TrainingData::getSingleBatches -> problem : trainingData not valid");
		}


		if (inputs.empty()) {
			throw std::runtime_error("TrainingData::getSingleBatches called on empty dataset.");
		}

		Matrix input_batch(inputs[0].size(), inputs.size());
		Matrix output_batch(outputs[0].size(), outputs.size());

		for (size_t col = 0; col < inputs.size(); ++col) {
			for (size_t row = 0; row < inputs[col].size(); ++row) {

				input_batch.setAt(inputs[col][row], row, col);

			}
			for (size_t row = 0; row < outputs[col].size(); ++row) {

				output_batch.setAt(outputs[col][row], row, col);
			}
		}

		

		return { DeviceMatrix(input_batch), DeviceMatrix(output_batch) };
	}

	std::pair<std::vector<DeviceMatrix>, std::vector<DeviceMatrix>> TrainingData::getMiniBatches(size_t batch_size) {

		if (this->inputs.empty())  // implies outputs are empty;  
			throw std::runtime_error("there is no inputs and outputs");
		
		size_t dataset_size = this->inputs.size();  // implies outputs.size();
		size_t input_dim = this->inputs[0].size();   
		size_t output_dim = this->outputs[0].size();

		if (batch_size > dataset_size)
			batch_size = dataset_size;

		size_t batches_count = (dataset_size  + batch_size - 1) / batch_size;


		std::vector<size_t> indices(dataset_size, 0);

		for (size_t i = 0; i < dataset_size; ++i) {
			indices[i] = i;
		}

		static std::random_device rd;
		static std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);


		std::vector<DeviceMatrix> input_batches;
		std::vector<DeviceMatrix> output_batches;

		input_batches.reserve(batches_count);
		output_batches.reserve(batches_count);

		for (size_t start = 0; start < dataset_size; start += batch_size) {  // start je batch iterator iterira po svim mini-batchevima

			size_t end = std::min(start + batch_size, dataset_size);  // kraj batcha
			size_t current_batch_size = end - start;  // velicina batcha

			Matrix input_batch(input_dim, current_batch_size);  // kreiranje input-mini batcha i output mini batcha
			Matrix output_batch(output_dim, current_batch_size);

			for (size_t col = 0; col < current_batch_size; ++col) {
				size_t idx = indices[start + col];  // idx iterira kroz sve sampleove mini-batcha na kojeg pointa start

				for (size_t row = 0; row < input_dim; ++row) {  // iteriranje kroz sve elemente svakog input -samplea
					input_batch.setAt(this->inputs[idx][row], row, col);
				}

				for (size_t row = 0; row < output_dim; ++row) {
					output_batch.setAt(this->outputs[idx][row], row, col);
				}
			}

			

			input_batches.push_back(DeviceMatrix(input_batch));
			output_batches.push_back(DeviceMatrix(output_batch));
		}


		return { input_batches, output_batches };
	}

};