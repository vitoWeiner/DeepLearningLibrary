// file: DeviceMatrix.cu, ovdje se nalaze metode za rad s DeviceMatrix preko hosta ali ne i preko devicea (nema kernela)


#include "../include/DeviceMatrix.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>  
#include <initializer_list>

namespace dl {


    DeviceMatrix::DeviceMatrix() :
        device_matrix(nullptr), rows_count(0), cols_count(0), total_size(0) {
      
    }

    DeviceMatrix::DeviceMatrix(const DeviceMatrix& other) :
        device_matrix(nullptr), rows_count(other.rows_count), cols_count(other.cols_count), total_size(other.total_size) {

        if (other.device_matrix == nullptr) {
            return;
        }

       

        cudaError_t cuda_malloc_error = cudaMalloc(&device_matrix, this->total_size * sizeof(float));

        if (cuda_malloc_error != cudaSuccess) {
            throw std::runtime_error("DeviceMatrix copy constructor error:\n cudaMalloc failed: " + std::string(cudaGetErrorString(cuda_malloc_error)));

        }

        cudaError_t cuda_memcpy_error = cudaMemcpy(device_matrix, other.device_matrix, this->total_size * sizeof(float), cudaMemcpyDeviceToDevice);

        if (cuda_memcpy_error != cudaSuccess) {
            cudaFree(this->device_matrix);
            throw std::runtime_error("DeviceMatrix copy constructor error:\n cudaMemcpy failed: " + std::string(cudaGetErrorString(cuda_memcpy_error)));
        }

     
    }

    DeviceMatrix& DeviceMatrix::operator=(const DeviceMatrix& other)
    {
        if (this == &other) {
            return *this;
        }

        if (this->device_matrix) {
            cudaError_t err = cudaFree(this->device_matrix);
            if (err != cudaSuccess) {
                
                printf("[DeviceMatrix] Warning: cudaFree failed in assignment: %s\n", cudaGetErrorString(err));
            }
            this->device_matrix = nullptr;
           
        }

        this->rows_count = other.rows_count;
        this->cols_count = other.cols_count;
        this->total_size = other.total_size;

        if (other.device_matrix == nullptr) {
            this->device_matrix = nullptr;
            return *this;
        }

       

        cudaError_t cuda_malloc_error = cudaMalloc(&this->device_matrix, this->total_size * sizeof(float));

        if (cuda_malloc_error != cudaSuccess) {
            throw std::runtime_error("DeviceMatrix assignment error:\n cudaMalloc failed: " + std::string(cudaGetErrorString(cuda_malloc_error)));
        }

        cudaError_t cuda_memcpy_error = cudaMemcpy(this->device_matrix, other.device_matrix, this->total_size * sizeof(float), cudaMemcpyDeviceToDevice);

        if (cuda_memcpy_error != cudaSuccess) {
            cudaFree(this->device_matrix);
            throw std::runtime_error("DeviceMatrix assignment error:\n cudaMemcpy failed: " + std::string(cudaGetErrorString(cuda_memcpy_error)));
        }

       

        return *this;
    }

    DeviceMatrix::DeviceMatrix(const Matrix& mat) :
        device_matrix(nullptr), rows_count(mat.rows()), cols_count(mat.cols()), total_size(mat.totalSize()) {

        const float* temp_arr = mat.borrowData();

        if (temp_arr == nullptr) {
            return;
        }

       

        cudaError_t cuda_malloc_error = cudaMalloc(&device_matrix, this->total_size * sizeof(float));

        if (cuda_malloc_error != cudaSuccess) {
            throw std::runtime_error("error from CUDA_Matrix constructor-0:\n cudaMalloc failed:\n " + std::string(cudaGetErrorString(cuda_malloc_error)));
        }

        cudaError_t cuda_memcpy_error = cudaMemcpy(device_matrix, temp_arr, this->total_size * sizeof(float), cudaMemcpyHostToDevice);

        if (cuda_memcpy_error != cudaSuccess) {

            cudaFree(device_matrix);
            throw std::runtime_error("error in CUDA_Matrix.uploadFromMatrix(Matrix M): problem:\n cudaMemcpy failed:\n" + std::string(cudaGetErrorString(cuda_memcpy_error)));
        }

      
    }

    DeviceMatrix::DeviceMatrix(std::initializer_list<float> args, size_t rows, size_t cols) :
        device_matrix(nullptr), rows_count(rows), cols_count(cols), total_size(rows* cols)
    {

        if (args.size() != this->total_size) {
            throw std::invalid_argument("DeviceMatrix constructor error: initializer list size does not match dimensions.");
        }

        if (this->total_size == 0) {
            return;
        }

     

        float* temp_arr = new float[this->total_size];

        std::copy(args.begin(), args.end(), temp_arr);

        cudaError_t cuda_malloc_error = cudaMalloc(&this->device_matrix, this->total_size * sizeof(float));

        if (cuda_malloc_error != cudaSuccess) {
            delete[] temp_arr;
            throw std::runtime_error("DeviceMatrix constructor error:\n cudaMalloc failed: " + std::string(cudaGetErrorString(cuda_malloc_error)));
        }

        cudaError_t cuda_memcpy_error = cudaMemcpy(this->device_matrix, temp_arr, this->total_size * sizeof(float), cudaMemcpyHostToDevice);

        delete[] temp_arr;

        if (cuda_memcpy_error != cudaSuccess) {
            cudaFree(this->device_matrix);
            throw std::runtime_error("DeviceMatrix constructor error:\n cudaMemcpy failed: " + std::string(cudaGetErrorString(cuda_memcpy_error)));
        }

       
    }



    DeviceMatrix::DeviceMatrix(size_t rows, size_t cols) :
        device_matrix(nullptr), rows_count(rows), cols_count(cols), total_size(rows* cols)
    {

        if (this->total_size == 0) {
            return;
        }

       

        cudaError_t cuda_malloc_error = cudaMalloc(&this->device_matrix, this->total_size * sizeof(float));

        if (cuda_malloc_error != cudaSuccess) {
            throw std::runtime_error("DeviceMatrix constructor error:\n cudaMalloc failed: " + std::string(cudaGetErrorString(cuda_malloc_error)));
        }

        cudaError_t cuda_memset_error = cudaMemset(this->device_matrix, 0, this->total_size * sizeof(float));

        if (cuda_memset_error != cudaSuccess) {
            cudaFree(this->device_matrix);
            throw std::runtime_error("DeviceMatrix constructor error:\n cudaMemset failed: " + std::string(cudaGetErrorString(cuda_memset_error)));
        }

       
    }

    

    DeviceMatrix::DeviceMatrix(DeviceMatrix&& other) noexcept
        : device_matrix(other.device_matrix),
        rows_count(other.rows_count),
        cols_count(other.cols_count),
        total_size(other.total_size)  
    {

      
        other.device_matrix = nullptr;
        other.rows_count = 0;
        other.cols_count = 0;
        other.total_size = 0;
    }



    DeviceMatrix& DeviceMatrix::operator=(DeviceMatrix&& other) noexcept { 

       

        if (this == &other) {
            return *this;
        }

        if (this->device_matrix) {
            cudaError_t err = cudaFree(this->device_matrix);
           

            if (err != cudaSuccess) {
                
            }
        }

        this->device_matrix = other.device_matrix;
        this->rows_count = other.rows_count;
        this->cols_count = other.cols_count;
        this->total_size = other.total_size;
       

        other.device_matrix = nullptr;
        other.rows_count = 0;
        other.cols_count = 0;
        other.total_size = 0;

        return *this;
    }




    size_t DeviceMatrix::rows() const noexcept {
        return this->rows_count;
    }

    size_t DeviceMatrix::cols() const noexcept {
        return this->cols_count;
    }

    size_t DeviceMatrix::totalSize() const noexcept {
        return this->total_size;
    }

    const float* DeviceMatrix::borrowData() const noexcept {
        return this->device_matrix;
    }


    Matrix DeviceMatrix::downloadToHost() const { 

        if (this->device_matrix == nullptr) {
            return Matrix(0, 0); 
        }

        float* host_data = new float[total_size];

        cudaError_t err = cudaMemcpy(host_data, device_matrix, total_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            delete[] host_data;
            throw std::runtime_error("DeviceMatrix constructor error:\n cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        }

        Matrix result(host_data, rows_count, cols_count);

        delete[] host_data;

        return result;
    }

    DeviceMatrix DeviceMatrix::matConcatCols(const DeviceMatrix& A, const DeviceMatrix& B) {

        if (A.rows() != B.rows())
            throw std::runtime_error("in function matConcatCols : runtime error : A.rows != B.rows");

        DeviceMatrix result(A.rows(), A.cols() + B.cols());


        // bytes_in_row_* = every row of matrix * has same number of elements -> same number of bytes, this is how much
        size_t bytes_in_row_A = A.cols() * sizeof(float);
        size_t bytes_in_row_B = B.cols() * sizeof(float);
        size_t bytes_in_row_result = result.cols() * sizeof(float);

        for (size_t row = 0; row < A.rows(); ++row) {

            cudaError_t err = cudaMemcpy(result.device_matrix + row * result.cols(), A.device_matrix + row * A.cols(), bytes_in_row_A, cudaMemcpyDeviceToDevice);

            if (err != cudaSuccess) {
                throw std::runtime_error("in function DeviceMatrix::matConcatCols cudaMemcpy failed : " + std::string(cudaGetErrorString(err)));
            }


            err = cudaMemcpy(result.device_matrix + row * result.cols() + A.cols(), B.device_matrix + row * B.cols(), bytes_in_row_B, cudaMemcpyDeviceToDevice);

            if (err != cudaSuccess) {
                throw std::runtime_error("in function DeviceMatrix::matConcatCols cudaMemcpy failed : " + std::string(cudaGetErrorString(err)));
            }
        }


        return result;
    }

    DeviceMatrix DeviceMatrix::matConcatRows(const DeviceMatrix& A, const DeviceMatrix& B) {
        if (A.cols() != B.cols())
            throw std::runtime_error("in function matConcatRows : runtime error : A.cols() != B.cols()");

        DeviceMatrix result(A.rows() + B.rows(), A.cols());

        cudaError_t err = cudaMemcpy(result.device_matrix, A.device_matrix, A.totalSize() * sizeof(float), cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
            throw std::runtime_error("in function DeviceMatrix::matConcatRows : cudaMemcpy failed : " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMemcpy(result.device_matrix + A.totalSize(), B.device_matrix, B.totalSize() * sizeof(float), cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
            throw std::runtime_error("in function DeviceMatrix::matConcatRows : cudaMemcpy failed : " + std::string(cudaGetErrorString(err)));
        }

        return result;
    }

void DeviceMatrix::downloadToHost(float* buffer) const {  // moram dodat error handling

    if (this->device_matrix == nullptr) {
        return;
    }



    cudaError_t err = cudaMemcpy(buffer, device_matrix, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {

        throw std::runtime_error("DeviceMatrix constructor error:\n cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    return;
}


void DeviceMatrix::clean() noexcept {
    if (this->device_matrix != nullptr) {

        cudaError_t err = cudaFree(this->device_matrix);
        if (err != cudaSuccess) {
            //std::cerr << "[DeviceMatrix] Warning: cudaFree failed in clean: " << cudaGetErrorString(err) << std::endl;
        }

        this->device_matrix = nullptr;
        this->rows_count = 0;
        this->cols_count = 0;
        this->total_size = 0;
	   
    }

}

DeviceMatrix::~DeviceMatrix() {
    if (device_matrix != nullptr) {
        cudaFree(device_matrix);

    }
}

};