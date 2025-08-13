#include "../../../include/Models/TrainingData/MLP_TrainingData.cuh"
#include <stdexcept>

namespace dl {
	TrainingData::TrainingData() {}

	void TrainingData::add(const std::vector<float>& input, const std::vector<float>& output) {
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




};