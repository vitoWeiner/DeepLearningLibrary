#pragma once


#include "Matrix.hpp"
#include <initializer_list>
#include <functional>
#include <stdexcept> // for std::runtime_error
#include <unordered_set>

namespace dl {

    class DeviceMatrix {
    
    protected:
        float* device_matrix;
        size_t rows_count;
        size_t cols_count;
        size_t total_size;
        
    public:
        
        
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
    	void clean() noexcept;  // this cleans up all data from object (dealocation)


        static DeviceMatrix matConcatCols(const DeviceMatrix& A, const DeviceMatrix& B);
        static DeviceMatrix matConcatRows(const DeviceMatrix& A, const DeviceMatrix& B);
         static DeviceMatrix matMul(const DeviceMatrix& A, const DeviceMatrix& B); 
       
        static DeviceMatrix matAdd(const DeviceMatrix& A, const DeviceMatrix& B); 
        static DeviceMatrix matSub(const DeviceMatrix& A, const DeviceMatrix& B); 
        static DeviceMatrix matElementWiseMul(const DeviceMatrix& A, const DeviceMatrix& B);
       
        static DeviceMatrix matTranspose(const DeviceMatrix& A);
        static DeviceMatrix matScale(const DeviceMatrix& A, float scalar);
        static DeviceMatrix matColSum(const DeviceMatrix& A); 
        static DeviceMatrix matRowSum(const DeviceMatrix& A); 
        static DeviceMatrix matColSumV2(const DeviceMatrix& A);
        static DeviceMatrix matRowSumV2(const DeviceMatrix& A);
       
        static DeviceMatrix trivialMatMul(const DeviceMatrix& A, const DeviceMatrix& B);
        static DeviceMatrix trivialUncoalescedMatMul(const DeviceMatrix& A, const DeviceMatrix& B);
    
        static DeviceMatrix Identity(size_t size);
        static DeviceMatrix Zero(size_t rows, size_t cols);
        static DeviceMatrix Random(size_t rows, size_t cols, std::pair<float, float> range = { 0.0f, 1.0f });
        static DeviceMatrix broadcastAdd(const DeviceMatrix& matrix, const DeviceMatrix& vector); 
    
        // activations
        static DeviceMatrix ReLU(const DeviceMatrix& input); 
        static DeviceMatrix ReLUGradient(const DeviceMatrix& input); 
        static DeviceMatrix Sigmoid(const DeviceMatrix& input);  
        static DeviceMatrix SigmoidGradient(const DeviceMatrix& sigmoid_output);
    
        // costs
        static DeviceMatrix MSE(const DeviceMatrix& output, const DeviceMatrix& target);  
        static DeviceMatrix MSEGradient(const DeviceMatrix& output, const DeviceMatrix& target);  


       static DeviceMatrix BCEGradient(const DeviceMatrix& output, const DeviceMatrix& target);
       static DeviceMatrix BCE(const DeviceMatrix& output, const DeviceMatrix& target);
    
        ~DeviceMatrix();

	}; // class DeviceMatrix

   
};  // namespace dl