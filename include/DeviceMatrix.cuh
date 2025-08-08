#pragma once


#include "Matrix.hpp"
#include <initializer_list>
#include <functional>

class DeviceMatrix {

protected:
    float* device_matrix;
    size_t rows_count;
    size_t cols_count;
    size_t total_size;

public:
    //DeviceMatrix(const DeviceMatrix&) = delete;
    //DeviceMatrix& operator=(const DeviceMatrix&) = delete;
    DeviceMatrix();
    DeviceMatrix(const DeviceMatrix& other);
    DeviceMatrix& operator=(const DeviceMatrix& other);

    DeviceMatrix(const Matrix& mat);
    DeviceMatrix(std::initializer_list<float> args, size_t rows, size_t cols);
    DeviceMatrix(size_t rows, size_t cols);
    DeviceMatrix(DeviceMatrix&& other) noexcept;
    DeviceMatrix& operator=(DeviceMatrix&& other) noexcept;

    size_t rows() const noexcept;
    size_t cols() const noexcept;
    size_t totalSize() const noexcept;
    const float* borrowData() const noexcept;

    Matrix downloadToHost() const;
    void downloadToHost(float* buffer) const;
	void clean() noexcept;

    static DeviceMatrix matMul(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
   
    static DeviceMatrix matAdd(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
    static DeviceMatrix matSub(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
    static DeviceMatrix matElementWiseMul(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
   
    static DeviceMatrix matTranspose(const DeviceMatrix& A); // working fine
    static DeviceMatrix matScale(const DeviceMatrix& A, float scalar); // working fine
    static DeviceMatrix matColReduce(const DeviceMatrix& A); // working fine
    static DeviceMatrix matRowReduce(const DeviceMatrix& A); // working fine
   
    static DeviceMatrix trivialMatMul(const DeviceMatrix& A, const DeviceMatrix& B);
    static DeviceMatrix trivialUncoalescedMatMul(const DeviceMatrix& A, const DeviceMatrix& B);

    static DeviceMatrix Identity(size_t size);
    static DeviceMatrix Zero(size_t rows, size_t cols);
    static DeviceMatrix Random(size_t rows, size_t cols, std::pair<float, float> range = { 0.0f, 1.0f });
    static DeviceMatrix broadcastAdd(const DeviceMatrix& matrix, const DeviceMatrix& vector); // working fine

    // aktivacije
    static DeviceMatrix ReLU(const DeviceMatrix& input); // working fine
    static DeviceMatrix ReLUGradient(const DeviceMatrix& input); // working fine
    static DeviceMatrix Sigmoid(const DeviceMatrix& input);  // working fine
    static DeviceMatrix SigmoidGradient(const DeviceMatrix& sigmoid_output); // working fine

    // costs
    static DeviceMatrix MSE(const DeviceMatrix& output, const DeviceMatrix& target);  // ovo nije bitno dal radi dobro ili ne jer se ne koristi direktno za treniranje vec cisto kao pokazatelj
    static DeviceMatrix MSEGradient(const DeviceMatrix& output, const DeviceMatrix& target);  // works fine

    ~DeviceMatrix();
};

