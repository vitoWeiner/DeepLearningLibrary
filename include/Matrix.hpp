#pragma once


#include <initializer_list>
#include <utility>      // std::pair
#include <string>       // std::string
#include <vector>

class Matrix {
protected:
    float* matrix;
    size_t rows_count;
    size_t cols_count;
    size_t total_size;

public:
    explicit Matrix();
    explicit Matrix(float* args, size_t rows, size_t cols);
    explicit Matrix(std::initializer_list<float> args, size_t rows, size_t cols);
    explicit Matrix(const std::vector<float>& args, size_t rows, size_t cols);
    explicit Matrix(size_t rows, size_t cols);

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other) noexcept;

    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    friend void swap(Matrix& a, Matrix& b) noexcept;

    void setAt(float value, size_t row, size_t col);
    void setAt(float value, size_t global_index);

    float getAt(size_t row, size_t col) const;
    float getAt(size_t global_index) const;

    void print(size_t rows = 0, size_t cols = 0) const noexcept;

    size_t rows() const noexcept;
    size_t cols() const noexcept;
    size_t totalSize() const noexcept;

    const float* borrowData() const noexcept;

    Matrix downloadToBinary(const std::string& filename) const;

    Matrix transpose() const;

	void clean() noexcept;


    // statics:

    static Matrix Identity(size_t size);
    static Matrix Zero(size_t rows, size_t cols);
    static Matrix Random(size_t rows, size_t cols, std::pair<float, float> range = { 0.0f, 1.0f });
    static Matrix DownloadFromBinary(const std::string& filename);

    static Matrix elementWiseMultiply(const Matrix& a, const Matrix& b);
    static Matrix columnReduce(const Matrix& a);
    static Matrix rowReduce(const Matrix& a);

    static Matrix relu(const Matrix& a);
    static Matrix reluGradient(const Matrix& a);
    static Matrix sigmoid(const Matrix& a);
    static Matrix sigmoidGradient(const Matrix& a);
    static Matrix broadcastAdd(const Matrix& a, const Matrix& b);
    static Matrix MSEGradient(const Matrix& a, const Matrix& t);

    ~Matrix() noexcept;
};

Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, float scalar);
Matrix operator*(float scalar, const Matrix& a);

bool operator==(const Matrix& a, const Matrix& b);
bool operator!=(const Matrix& a, const Matrix& b);