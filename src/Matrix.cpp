#include "../include/Matrix.hpp"

#include <cstring>      // std::memcpy
#include <stdexcept>    // std::invalid_argument
#include <algorithm>    // std::copy
#include <cstdio>       // printf
#include <fstream>      // std::ofstream
#include <cmath>        // std::fabs


namespace dl {


Matrix::Matrix() : rows_count(0), cols_count(0), total_size(0), matrix(nullptr) {}

Matrix::Matrix(float* args, size_t rows, size_t cols)
    : rows_count(rows), cols_count(cols)

{

    this->total_size = rows * cols;

    if (this->total_size == 0) {
        this->matrix = nullptr;
        return;
    }

    if (!args) {
        throw std::invalid_argument("Matrix constructor error : input array is null.");
    }

    this->matrix = new float[this->total_size];

    std::memcpy(this->matrix, args, this->total_size * sizeof(float));

}

Matrix::Matrix(std::initializer_list<float> args, size_t rows, size_t cols) : rows_count(rows), cols_count(cols)
{

    this->total_size = rows * cols;

    if (args.size() != this->total_size) {
        throw std::invalid_argument("Matrix constructor error : initializer list size does not match matrix dimensions");
    }

    if (this->total_size == 0) {
        this->matrix = nullptr;
        return;
    }

    this->matrix = new float[this->total_size];
    std::copy(args.begin(), args.end(), this->matrix);
}

Matrix::Matrix(const std::vector<float>& args, size_t rows, size_t cols) : rows_count(rows), cols_count(cols), total_size(rows* cols) {

    if (total_size != args.size()) {
        throw std::runtime_error("In matrix constructor, dimensions are not same as vector size");
    }

    this->matrix = new float[total_size];

    std::copy(args.begin(), args.end(), this->matrix);

}

Matrix::Matrix(size_t rows, size_t cols) : rows_count(rows), cols_count(cols) {


    this->total_size = rows * cols;

    this->matrix = nullptr;

    if (this->total_size != 0) {
        matrix = new float[this->total_size](); 
    }

}

Matrix::Matrix(const Matrix& other) : rows_count(other.rows_count), cols_count(other.cols_count), total_size(other.total_size) {

    if (other.matrix == nullptr) {
        this->matrix = nullptr;
        return;
    }

    this->matrix = new float[this->total_size];
    std::memcpy(this->matrix, other.matrix, this->total_size * sizeof(float));

}

void swap(Matrix& a, Matrix& b) noexcept {
    using std::swap;
    swap(a.matrix, b.matrix);
    swap(a.rows_count, b.rows_count);
    swap(a.cols_count, b.cols_count);
    swap(a.total_size, b.total_size);
}


Matrix& Matrix::operator=(const Matrix& other) noexcept {



    if (this == &other) return *this;

    Matrix temp(other);
    swap(*this, temp);

    return *this;

}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_count(other.rows_count), cols_count(other.cols_count),
    total_size(other.total_size), matrix(other.matrix) {

   other.matrix = nullptr;
   other.total_size = 0;
   other.rows_count = 0;
   other.cols_count = 0;
}


Matrix& Matrix::operator=(Matrix&& other) noexcept {

    if (this == &other)
        return *this;

    if (this->matrix != nullptr)
        delete[] this->matrix;

    this->matrix = other.matrix;
    this->rows_count = other.rows_count;
    this->cols_count = other.cols_count;
    this->total_size = other.total_size;

    other.matrix = nullptr;
    other.total_size = 0;
    other.rows_count = 0;
    other.cols_count = 0;

    return *this;

}



bool operator==(const Matrix& a, const Matrix& b) {
    if (
        a.rows() != b.rows() ||
        a.cols() != b.cols())
        return false;

    const float epsilon = 0.1;

    size_t rows = a.rows();
    size_t cols = a.cols();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            float val_a = a.getAt(row, col);
            float val_b = b.getAt(row, col);

            if (std::fabs(val_a - val_b) > epsilon) {
                printf("\nstopper : %f, %f\n", val_a, val_b);
                return false;
            }
        }
    }

    return true;
}

bool operator!=(const Matrix& a, const Matrix& b) {
    return !operator==(a, b);
}



void Matrix::setAt(float value, size_t row, size_t col) {

    if (row >= this->rows_count) {
        throw std::invalid_argument("Matrix setAt error : indexed row is >= number of rows in matrix");
    }

    if (col >= this->cols_count) {
        throw std::invalid_argument("Matrix setAt error : indexed col is >= number of columns in matrix");
    }

    size_t global_idx = row * this->cols_count + col;
    this->matrix[global_idx] = value;

}

void Matrix::setAt(float value, size_t global_index) {

    if (global_index >= this->total_size) {
        throw std::invalid_argument("Matrix setAt error : indexed row is >= number of rows in matrix");
    }

    this->matrix[global_index] = value;
}

float Matrix::getAt(size_t row, size_t col) const {
    if (row >= this->rows_count) {
        throw std::invalid_argument("Matrix setAt error : indexed row is >= number of rows in matrix");
    }

    if (col >= this->cols_count) {
        throw std::invalid_argument("Matrix setAt error : indexed col is >= number of columns in matrix");
    }

    size_t global_idx = row * this->cols_count + col;

    return this->matrix[global_idx];
}

bool Matrix::check(const std::function<bool(float)>& predicate) {
    for (size_t row = 0; row < this->rows(); ++row) {
        for (size_t col = 0; col < this->cols(); ++col) {
            if (!predicate(this->getAt(row, col)))
                return false;
        }
    }

    return true;
}

Matrix Matrix::matConcatCols(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows())
        throw std::runtime_error("in Matrix::matConcatCols : a.rows() != b.rows()");

    Matrix result(a.rows(), a.cols() + b.cols());

    for (size_t row = 0; row < result.rows(); ++row) {
        std::memcpy(result.matrix + row * result.cols(), a.matrix + row * a.cols(), sizeof(float) * a.cols());
        std::memcpy(result.matrix + row * result.cols() + a.cols(), b.matrix + row * b.cols(), sizeof(float) * b.cols());
    }

    return result;

}

Matrix Matrix::matConcatRows(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.cols())
        throw std::runtime_error("in Matrix::matConcatRows : a.cols() != b.cols()");

    Matrix result(a.rows() + b.rows(), a.cols());

    std::memcpy(result.matrix, a.matrix, a.totalSize() * sizeof(float));
    std::memcpy(result.matrix + a.totalSize(), b.matrix, b.totalSize() * sizeof(float));

    return result;

}

Matrix Matrix::elementWiseMultiply(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Matrix elementWiseMultiply error: incompatible dimensions");
    }
    Matrix result(a.rows(), a.cols());
    for (size_t row = 0; row < a.rows(); ++row) {
        for (size_t col = 0; col < a.cols(); ++col) {
            result.setAt(a.getAt(row, col) * b.getAt(row, col), row, col);
        }
    }
    return result;
}


float Matrix::getAt(size_t global_index) const {

    if (global_index >= this->total_size) {
        throw std::invalid_argument("Matrix setAt error : indexed row is >= number of rows in matrix");
    }

    return this->matrix[global_index];

}


std::vector<float> Matrix::toStdVector() {

    std::vector<float> result;
    size_t size = this->totalSize();

    result.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        result.push_back(this->matrix[i]);
    }

    return result;
    
}


void Matrix::print(size_t rows, size_t cols) const noexcept {

    if (rows == 0) {
        rows = rows_count;
    }

    if (cols == 0) {
        cols = cols_count;
    }

    if (rows > rows_count)
        rows = rows_count;

    if (cols > cols_count)
        cols = cols_count;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            printf("%.4f ", this->matrix[row * cols_count + col]);
        }
        printf("\n");
    }
}

size_t Matrix::rows() const noexcept {

    return this->rows_count;
}

size_t Matrix::cols() const noexcept {

    return this->cols_count;
}

size_t Matrix::totalSize() const noexcept {

    return this->total_size;
}

const float* Matrix::borrowData() const noexcept {
    return this->matrix;
}


Matrix::~Matrix() noexcept {
    if (this->matrix != nullptr) {
        delete[] this->matrix;
    }
}


Matrix operator+(const Matrix& a, const Matrix& b) {
    Matrix result(a.rows(), a.cols());

    for (size_t row = 0; row < a.rows(); ++row) {
        for (size_t col = 0; col < a.cols(); ++col) {
            result.setAt(a.getAt(row, col) + b.getAt(row, col), row, col);
        }
    }

    return result;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    Matrix result(a.rows(), a.cols());
    for (size_t row = 0; row < a.rows(); ++row) {
        for (size_t col = 0; col < a.cols(); ++col) {
            result.setAt(a.getAt(row, col) - b.getAt(row, col), row, col);
        }
    }
    return result;
}

Matrix operator*(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Matrix multiplication error: incompatible dimensions");
    }
    Matrix result(a.rows(), b.cols());
    for (size_t row = 0; row < a.rows(); ++row) {
        for (size_t col = 0; col < b.cols(); ++col) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols(); ++k) {
                sum += a.getAt(row, k) * b.getAt(k, col);
            }
            result.setAt(sum, row, col);
        }
    }
    return result;
}

Matrix operator*(const Matrix& a, float scalar) {
    Matrix result(a.rows(), a.cols());
    for (size_t row = 0; row < a.rows(); ++row) {
        for (size_t col = 0; col < a.cols(); ++col) {
            result.setAt(a.getAt(row, col) * scalar, row, col);
        }
    }
    return result;
}

Matrix operator*(float scalar, const Matrix& a) {
    return a * scalar;
}


Matrix Matrix::columnReduce(const Matrix& a) {
    Matrix result(a.rows(), 1);

    for (size_t row = 0; row < a.rows(); ++row) {

        float sum = 0.0f;

        for (size_t col = 0; col < a.cols(); ++col) {
            sum += a.getAt(row, col);
        }

        result.setAt(sum, row, 0);
    }

    return result;
}


Matrix Matrix::rowReduce(const Matrix& a) {

    Matrix result(1, a.cols());

    for (size_t col = 0; col < a.cols(); ++col) {

        float sum = 0.0f;

        for (size_t row = 0; row < a.rows(); ++row) {
            sum += a.getAt(row, col);
        }

        result.setAt(sum, 0, col);
    }

    return result;

}

// deep learning features : start

Matrix Matrix::relu(const Matrix& input) {
    Matrix output(input.rows(), input.cols());

    for (size_t row = 0; row < input.rows(); ++row) {
        for (size_t col = 0; col < input.cols(); ++col) {
            float value = input.getAt(row, col);
            output.setAt(value > 0 ? value : 0, row, col);
        }
    }

    return output;
}

Matrix Matrix::reluGradient(const Matrix& input) {

    Matrix output(input.rows(), input.cols());

    for (size_t row = 0; row < input.rows(); ++row) {
        for (size_t col = 0; col < input.cols(); ++col) {
            float value = input.getAt(row, col);
            if (value > 0) {
                output.setAt(1, row, col);
            }
            else {
                output.setAt(0, row, col);
            }
        }
    }

    return output;

}

Matrix Matrix::sigmoid(const Matrix& input) {
    Matrix output(input.rows(), input.cols());

    for (size_t row = 0; row < input.rows(); ++row) {
        for (size_t col = 0; col < input.cols(); ++col) {
            float value = input.getAt(row, col);
            output.setAt(1.0f / (1.0f + std::exp(-value)), row, col);
        }
    }

    return output;
}

Matrix Matrix::sigmoidGradient(const Matrix& sigmoidOutput) {



    Matrix output(sigmoidOutput.rows(), sigmoidOutput.cols());

    for (size_t row = 0; row < sigmoidOutput.rows(); ++row) {
        for (size_t col = 0; col < sigmoidOutput.cols(); ++col) {

            float sigma = sigmoidOutput.getAt(row, col);
            output.setAt(sigma * (1 - sigma), row, col);
        }
    }

    return output;

}

Matrix Matrix::MSEGradient(const Matrix& a, const Matrix& t) {
    if (a.totalSize() != t.totalSize())
        printf("shit happend");

    Matrix output(a.rows(), a.cols());

    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {

            float value = a.getAt(i, j) - t.getAt(i, j);

            value /= (a.rows() * a.cols());

            value *= 2;

            output.setAt(value, i, j);
        }
    }

    return output;

}

Matrix Matrix::broadcastAdd(const Matrix& a, const Matrix& b) {

    if (a.cols() != 1 || a.rows() != b.rows())
        printf("shit happend");

    Matrix output(b.rows(), b.cols());

    size_t rows = a.rows();

    for (size_t col = 0; col < b.cols(); ++col) {
        for (size_t row = 0; row < rows; ++row) {
            output.setAt(a.getAt(row, 0) + b.getAt(row, col), row, col);

        }
    }

    return output;
}


// deep learning features : end

Matrix Matrix::Identity(size_t size) {

    Matrix identity(size, size);

    for (size_t i = 0; i < size; ++i) {
        identity.setAt(1.0f, i, i);
    }

    return identity;
}

Matrix Matrix::transpose() const {
    Matrix transposed(this->cols(), this->rows());

    for (size_t row = 0; row < this->rows(); ++row) {
        for (size_t col = 0; col < this->cols(); ++col) {

            transposed.setAt(this->getAt(row, col), col, row);
        }
    }

    return transposed;
}

Matrix Matrix::Zero(size_t rows, size_t cols) {
    return Matrix(rows, cols);

}

Matrix Matrix::Random(size_t rows, size_t cols, std::pair<float, float> range) {
    Matrix random(rows, cols);

    float delta = range.second - range.first;


    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {

            float rand_normalized_float = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            float rand_value = range.first + rand_normalized_float * delta;

            random.setAt(rand_value, i, j);
        }
    }

    

    return random;
}




Matrix Matrix::downloadToBinary(const std::string& filename) const {

    std::ofstream ofile(filename, std::ios::binary);

    if (!ofile) {
        throw std::runtime_error("Matrix downloadToBinary error: could not open file for writing" + filename);
    }

    size_t rows = this->rows();
    size_t cols = this->cols();

    const float* data = this->borrowData();

    if (data == nullptr) {
        throw std::runtime_error("Matrix downloadToBinary error: matrix data is null");
    }

    ofile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    ofile.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));

    return *this;
}


Matrix Matrix::DownloadFromBinary(const std::string& filename) {

    std::ifstream ifile(filename, std::ios::binary);

    if (!ifile) {
        throw std::runtime_error("Matrix LoadFromBinary error: could not open file for reading " + filename);
    }

    size_t rows, cols;

    ifile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    Matrix result(rows, cols);

    ifile.read(reinterpret_cast<char*>(result.matrix), rows * cols * sizeof(float));

    if (!ifile) {
        throw std::runtime_error("Matrix LoadFromBinary error: could not read data from file " + filename);
    }

    return result;
}

void Matrix::clean() noexcept {
    
    if (this->matrix != nullptr) {
        delete[] this->matrix;
        this->matrix = nullptr;

        this->rows_count = 0;
        this->cols_count = 0;
        this->total_size = 0;

    }

}

};