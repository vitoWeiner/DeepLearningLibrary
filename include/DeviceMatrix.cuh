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
        static long long instances;
        static std::unordered_set<long long> ids;
     
		static long long ID;
		long long id;  

        static void increment(DeviceMatrix* edit) {
            return;
            if (DeviceMatrix::instances >= 10000)
				throw std::runtime_error("DeviceMatrix instances limit reached. Possible memory leak or incorrect usage.");
			DeviceMatrix::instances++;
			edit->id = ID;
            ids.insert(ID);
            ID++;
        }

        static void decrement(int id_m) {
            return;
            if (DeviceMatrix::instances == 0)
                printf("shit happend");
			DeviceMatrix::instances--;
            ids.erase(id_m);
        }

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
    	void clean() noexcept;  // clean up all data from object


        static DeviceMatrix matConcatCols(const DeviceMatrix& A, const DeviceMatrix& B);
        static DeviceMatrix matConcatRows(const DeviceMatrix& A, const DeviceMatrix& B);
         static DeviceMatrix matMul(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
       
        static DeviceMatrix matAdd(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
        static DeviceMatrix matSub(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
        static DeviceMatrix matElementWiseMul(const DeviceMatrix& A, const DeviceMatrix& B); // working fine
       
        static DeviceMatrix matTranspose(const DeviceMatrix& A); // working fine
        static DeviceMatrix matScale(const DeviceMatrix& A, float scalar); // working fine
        static DeviceMatrix matColSum(const DeviceMatrix& A); // working fine
        static DeviceMatrix matRowSum(const DeviceMatrix& A); // working fine
        static DeviceMatrix matColSumV2(const DeviceMatrix& A);
        static DeviceMatrix matRowSumV2(const DeviceMatrix& A);
       
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


       static DeviceMatrix BCEGradient(const DeviceMatrix& output, const DeviceMatrix& target);
       static DeviceMatrix DeviceMatrix::BCE(const DeviceMatrix& output, const DeviceMatrix& target);
    
        ~DeviceMatrix();

	}; // class DeviceMatrix

   
};  // namespace dl