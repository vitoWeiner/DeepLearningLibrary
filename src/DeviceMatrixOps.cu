// file: DeviceMatrixOps.cu, ovdje se nalaze metode za rad s DeviceMatrix preko devicea (kerneli)


/*

POZOR:

NEKI KERNELI NISU JOS OPTIMIZIRANI NEGO TRIVIJALNO IMPLEMENTIRANI, SEMANTICKI ISPRAVNI, RADE ISPRAVNO, ALI POTPUNA OPTIMIZACIJA JOS NIJE IZVEDENA.
NPR:
- reduction sum radi ali treba dodati umjesto trivijalnog algoritma, algoritam redukcije putem stabla redukcije, a ne trivijalno zbrajanje svakog retka.
- matmul je dosta optimiziran, radi ispravno, no postoje izvori sporosti poput bank konflikata, to jos treba rjesiti, neki problemi s bank konfliktima i mozda dodati jos koje optimizacije.
- banchmarking i optimiziranje velicine blokova i grida treba isto, trenutno su simbolicni 2x2, u praksi je cesto 16x16, ali treba testirati na uredaju i vidjeti sto je najoptimalnije.

*/


#include "./../include/DeviceMatrix.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdexcept>
#include <cstring>
#include <iostream>


#define TILE_WIDTH 16
#define TILE_HEIGHT 16


namespace dl {

__global__ void tiled_matmul(float* res, float* X, float* Y, int rowsX, int colsX, int colsY) {

    //assert(TILE_WIDTH == blockDim.x && TILE_HEIGHT == blockDim.y && TILE_WIDTH == TILE_HEIGHT);

    const int block_idx_row = blockDim.y * blockIdx.y; // indeks na prvi thread bloka (redak)
    const int block_idx_col = blockDim.x * blockIdx.x; // indeks na prvi thread bloka (stupac)

    int tiles_vector_size = (colsX + TILE_WIDTH - 1) / TILE_WIDTH; // koliko tileova sadrzi tile row od X odnosno tile col od Y. to je ceil(colsX / TILE_WIDTH)
    // jer imamo situaciju ceil(a, b) = (a + b - 1) / b; ovo sljedi iz toga sto 1.) ako je a djeljiv s b, a+b-1 nece preci sljedeci cijeli broj te ce rezultat djeljenja biti isti. No ako je ostatak bar 1, preci ce sljedeci cjeli broj i nece biti isti.

    __shared__ float sh_tile_x[TILE_HEIGHT][TILE_WIDTH];  // alociranje shared memorije na L1 cache za tile matrice X  (+ 1 radi bank konflikta)
    __shared__ float sh_tile_y[TILE_HEIGHT][TILE_WIDTH];  // alociranje shared memorije na L1 cache za tile matrice Y (+ 1 radi bank konflikta)

    float sum = 0.0f;

    for (int tile_iterator = 0; tile_iterator < tiles_vector_size; ++tile_iterator) {

        // svaki blok pristupa tile redu matrice X i tile stupcu matrice Y koji su jednaki tile_row i tile_col izlazne matrice (block_idx_row, block_idx_col)
        // da bi se izracunalo kojem tileu unutar redka X i kojem tileu unutar stupca Y u trenutnoj iteraciji se pristupa koristi se tile_iterator

        int X_tile_row = block_idx_row;
        int Y_tile_col = block_idx_col;

        int X_tile_col = tile_iterator * TILE_WIDTH;
        int Y_tile_row = tile_iterator * TILE_HEIGHT;

        // sada znamo row i col koordinate prvih elemenata unutar tileova redka X i stupca Y.

        int X_element_row = X_tile_row + threadIdx.y;
        int X_element_col = X_tile_col + threadIdx.x;
        int Y_element_row = Y_tile_row + threadIdx.y;
        int Y_element_col = Y_tile_col + threadIdx.x;


        if (X_element_row < rowsX && X_element_col < colsX) {
            sh_tile_x[threadIdx.y][threadIdx.x] = X[X_element_row * colsX + X_element_col]; //d_A_ptr[(row)*A_n_cols + (phase * TILE_WIDTH + tx)];
        }
        else {
            sh_tile_x[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (Y_element_row < colsX && Y_element_col < colsY) {
            sh_tile_y[threadIdx.y][threadIdx.x] = Y[Y_element_row * colsY + Y_element_col]; //d_B_ptr[(phase * TILE_WIDTH + ty) * C_n_cols + (col)];
        }
        else {
            sh_tile_y[threadIdx.y][threadIdx.x] = 0.0f;
        }


        __syncthreads();

        // u ovom trenutku su tileovi ucitani u shared memoriju, vrijeme je za matmul tileova.

        for (int element_iterator = 0; element_iterator < TILE_WIDTH; ++element_iterator) {

            sum += sh_tile_x[threadIdx.y][element_iterator] * sh_tile_y[element_iterator][threadIdx.x];

        }

        __syncthreads();

    }

    int row = block_idx_row + threadIdx.y;
    int col = block_idx_col + threadIdx.x;

    if (row < rowsX && col < colsY) {

        res[row * colsY + col] = sum;

    }

    return;

}


DeviceMatrix DeviceMatrix::matMul(const DeviceMatrix& A, const DeviceMatrix& B) {

    if (A.cols() != B.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication.");
    }

    if (A.totalSize() == 0 || B.totalSize() == 0) {
        throw std::runtime_error("One of the matrices in matMul is empty.");
    }

    size_t rows = A.rows();
    size_t cols = B.cols();
    size_t shared_dim = A.cols();

    DeviceMatrix result(rows, cols);

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);  // blok mora biti dimenzija istih kao i tileovi, dakle TILE_WIDTH x TILE_HEIGHT
    dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_HEIGHT - 1) / TILE_HEIGHT);  // dimenzije grida odreduju broj blokova u sirinu i visinu, a to je broj stupaca i redaka podijeljen s TILE_WIDTH odnosno TILE_HEIGHT. Koristi se ceil(a, b) = (a + b - 1) / b da bi dobili cijeli broj blokova ako nisu djeljivi (bolje visak pa imamo prazne threadove, nego ne pokriti cijeli).

    tiled_matmul << <dimGrid, dimBlock >> > (
        result.device_matrix,
        A.device_matrix,
        B.device_matrix,
        rows,
        shared_dim,
        cols
        );





    cudaError_t err = cudaGetLastError();

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;

}



__global__ void trivial_matmul_kernel(float* result, float* X, float* Y, int rowsX, int colsX, int colsY) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rowsX || col >= colsY)
        return;

    float sum = 0.0f;

    for (int el = 0; el < colsX; ++el) {
        sum += X[row * colsX + el] * Y[el * colsY + col];
    }

    result[row * colsY + col] = sum;
}



DeviceMatrix DeviceMatrix::trivialMatMul(const DeviceMatrix& A, const DeviceMatrix& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication.");
    }

    if (A.totalSize() == 0 || B.totalSize() == 0) {
        throw std::runtime_error("One of the matrices in matMul is empty.");
    }

    size_t rows = A.rows();
    size_t cols = B.cols();
    size_t shared_dim = A.cols();

    DeviceMatrix result(rows, cols);

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);  // blok mora biti dimenzija istih kao i tileovi, dakle TILE_WIDTH x TILE_HEIGHT
    dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_HEIGHT - 1) / TILE_HEIGHT);

    trivial_matmul_kernel << <dimGrid, dimBlock >> > (
        result.device_matrix,
        A.device_matrix,
        B.device_matrix,
        rows,
        shared_dim,
        cols
        );



    cudaError_t err = cudaGetLastError();

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}



__global__ void trivial_uncoalesced_matmul_kernel(float* result, float* X, float* Y, int rowsX, int colsX, int colsY) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rowsX && col >= colsY)
        return;

    float sum = 0.0f;

    for (int el = 0; el < colsX; ++el) {
        sum += X[row * colsX + el] * Y[el * colsY + col];
    }

    result[row * colsY + col] = sum;
}

DeviceMatrix DeviceMatrix::trivialUncoalescedMatMul(const DeviceMatrix& A, const DeviceMatrix& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication.");
    }

    if (A.totalSize() == 0 || B.totalSize() == 0) {
        throw std::runtime_error("One of the matrices in matMul is empty.");
    }

    size_t rows = A.rows();
    size_t cols = B.cols();
    size_t shared_dim = A.cols();

    DeviceMatrix result(rows, cols);

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);  // blok mora biti dimenzija istih kao i tileovi, dakle TILE_WIDTH x TILE_HEIGHT
    dim3 dimGrid((rows + TILE_WIDTH - 1) / TILE_WIDTH, (cols + TILE_HEIGHT - 1) / TILE_HEIGHT);

    trivial_uncoalesced_matmul_kernel << <dimGrid, dimBlock >> > (
        result.device_matrix,
        A.device_matrix,
        B.device_matrix,
        rows,
        shared_dim,
        cols
        );



    cudaError_t err = cudaGetLastError();

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}




// Zbrajanje matrica

__global__ void matAdd_kernel(float* result, const float* A, const float* B, size_t rows, size_t cols) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    result[idx] = A[idx] + B[idx];

}

__global__ void matSub_kernel(float* result, const float* A, const float* B, size_t rows, size_t cols) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    result[idx] = A[idx] - B[idx];
}

DeviceMatrix DeviceMatrix::matAdd(const DeviceMatrix& A, const DeviceMatrix& B) {

    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("Matrices must have the same dimensions for addition.");
    }

    if (A.totalSize() == 0 || B.totalSize() == 0) {
        throw std::runtime_error("One of the matrices in matAdd is empty.");
    }

    DeviceMatrix result(A.rows(), A.cols());

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((A.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (A.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    size_t size = A.rows() * A.cols();

    matAdd_kernel << <dimGrid, dimBlock >> > (result.device_matrix, A.device_matrix, B.device_matrix, A.rows(), A.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;

}

DeviceMatrix DeviceMatrix::matSub(const DeviceMatrix& A, const DeviceMatrix& B) {

    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("Matrices must have the same dimensions for subtraction.");
    }

    if (A.totalSize() == 0 || B.totalSize() == 0) {
        throw std::runtime_error("One of the matrices in matSub is empty.");
    }

    DeviceMatrix result(A.rows(), A.cols());
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((A.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (A.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);
    matSub_kernel << <dimGrid, dimBlock >> > (result.device_matrix, A.device_matrix, B.device_matrix, A.rows(), A.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;

}

// element-wise mnozenje matrica

__global__ void elementWiseMultiply_kernel(float* result, const float* A, const float* B, size_t rows, size_t cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;
    result[idx] = A[idx] * B[idx];

}

DeviceMatrix DeviceMatrix::matElementWiseMul(const DeviceMatrix& A, const DeviceMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("Matrices must have the same dimensions for element-wise multiplication.");
    }

    if (A.totalSize() == 0 || B.totalSize() == 0) {
        throw std::runtime_error("One of the matrices in matElementWiseMul is empty.");
    }

    DeviceMatrix result(A.rows(), A.cols());

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);

    dim3 dimGrid((A.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (A.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    elementWiseMultiply_kernel << <dimGrid, dimBlock >> > (result.device_matrix, A.device_matrix, B.device_matrix, A.rows(), A.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

// Transponiranje matrice




__global__ void tiled_transpose(float* res, float* X, int rowsX, int colsX) {

    __shared__ float sh_tile[TILE_HEIGHT][TILE_WIDTH + 1]; 

    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  


    if (row < rowsX && col < colsX) {
        sh_tile[threadIdx.y][threadIdx.x] = X[row * colsX + col];
    }

    __syncthreads();


    int transposed_row = blockIdx.x * blockDim.x + threadIdx.y;
    int transposed_col = blockIdx.y * blockDim.y + threadIdx.x;


    if (transposed_row < colsX && transposed_col < rowsX) {
        res[transposed_row * rowsX + transposed_col] = sh_tile[threadIdx.x][threadIdx.y];
    }
}

DeviceMatrix DeviceMatrix::matTranspose(const DeviceMatrix& A) {

    if (A.totalSize() == 0) {
        return A;
    }

    DeviceMatrix output(A.cols(), A.rows());
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((A.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (A.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    tiled_transpose << <dimGrid, dimBlock >> > (
        output.device_matrix,
        A.device_matrix,
        A.rows(),
        A.cols()
        );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}


// scalar multiplication

__global__ void scaling_kernel(float* result, const float* matrix, float scalar, size_t rows, size_t cols)
{

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    result[idx] = matrix[idx] * scalar;
}

DeviceMatrix DeviceMatrix::matScale(const DeviceMatrix& A, float scalar) {

    if (A.totalSize() == 0) {
        return A;
    }

    DeviceMatrix result(A.rows(), A.cols());

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((A.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (A.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    scaling_kernel << <dimGrid, dimBlock >> > (result.device_matrix, A.device_matrix, scalar, A.rows(), A.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

// sum-reduce trivial kernel

__global__ void sum_reduce_kernel(float* result, const float* matrix, size_t rows, size_t cols) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    if (row >= rows)
        return;

    for (size_t col = 0; col < cols; ++col) {
        sum += matrix[row * cols + col];
    }

    result[row] = sum;
}

DeviceMatrix DeviceMatrix::matColSum(const DeviceMatrix& A) {
    if (A.totalSize() == 0) {
        return A;
    }

    DeviceMatrix result(A.rows(), 1); // rezultat je vektor sa zbrojem svakog retka

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((result.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (result.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    sum_reduce_kernel << <dimGrid, dimBlock >> > (result.device_matrix, A.device_matrix, A.rows(), A.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
}


__inline__ __device__
float warp_reduce_sum(float val) {  // binary halving reduction


    unsigned mask = 0xffffffffu;  // mask of threads that participate in collective operation, ovdje sve pise : "https://developer.nvidia.com/blog/using-cuda-warp-level-primitives"


    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}

__global__
void col_reduce_op(float* result, const float* matrix, size_t rows, size_t cols, size_t pitch_cols) {

    size_t row = blockIdx.y;
    if (row >= rows) return;

    extern __shared__ float s_warp_sums[];


    float local = 0.0f;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {   // matrica moze imati vise stupaca nego je alociranih threadova, threadovi coalesced iteriraju po svim segmentima u periodu velicine blockDim.x
        size_t idx = row * pitch_cols + col;
        local += matrix[idx];  
    } // u ovom trenutku svi threadovi u warpu imaju svoju vrijednost u registru local

    


    local = warp_reduce_sum(local);  // nakon ove naredbe prvi thread warpa ce imati u registru local vrijednost koja predstavlja sumu za warp


    int lane = threadIdx.x & 31;  // prvi thrad warpa
    int warpId = threadIdx.x >> 5;  // indeks warpa
    if (lane == 0) s_warp_sums[warpId] = local;  // prvi thread warpa upisuje vrijednost sume za warp u shared memoriju
    __syncthreads();  // svi warpovi bloka moraju doci do ove linije prije dalje

    // jedan blok == jedan row matrice
    float block_sum = 0.0f;  // sada treba napraviti redukciju s_warp_sums polja da bi dobili za cijeli blok
    int numWarps = (blockDim.x + 31) / 32;  // broj warpova bloka
    if (threadIdx.x < 32) {  // prvi thread svakog warpa racuna redukciju s_warp_sums polja
        float v = (threadIdx.x < numWarps) ? s_warp_sums[threadIdx.x] : 0.0f;  // svaki thread warpa koji racuna redukciju pristupa sumi jednog warpa, ako je thread izvan warpa, postavlja 0.0f
        block_sum = warp_reduce_sum(v); // redukcija
    }


    if (threadIdx.x == 0) result[row] = block_sum;  // u svaki redak matrice result se zapisuje vrijednost za blok
}


DeviceMatrix DeviceMatrix::matColSumV2(const DeviceMatrix& A) {

    if (A.totalSize() == 0 || A.cols() == 1)
        return A;


    DeviceMatrix result(A.rows(), 1);

    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid(1, A.rows(), 1);
    size_t shared_bytes = sizeof(float) * ((threads + 31) / 32);  // == broj warpova u bloku * sizeof(float); // ako ce se staticki alocirati onda najbolje 32 warpa jer (blockDim.x <= 1024 --> warps count <= 32) ali ovako je bolje dinamicki alocirati da ne bude bezveze previse shared memory alocirano

    col_reduce_op << <grid, block, shared_bytes >> > (result.device_matrix, A.device_matrix, A.rows(), A.cols(), A.cols());  // kasnije mozda bude trebalo pitched cols

    cudaDeviceSynchronize();

    return result;

}


template<int ROWS_PER_TILE = 256>
__global__
void row_sum_op(float* result, const float* matrix, size_t rows, size_t cols) {

    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= cols)
        return;

    size_t row_start = blockIdx.y * ROWS_PER_TILE;
    size_t row_end = min(rows, row_start + ROWS_PER_TILE);

    float acc = 0.0f;

    for (size_t row = row_start; row < row_end; ++row) {
        acc += matrix[row * cols + col];
    }

    atomicAdd(&result[col], acc);

}

DeviceMatrix DeviceMatrix::matRowSumV2(const DeviceMatrix& A) {

    int threads = 256;
    const int rows_per_tile = 256;

    dim3 block(threads, 1, 1);
    dim3 grid((A.cols() + threads - 1) / threads,
              (A.rows() + rows_per_tile - 1 ) / rows_per_tile,
              1);

    DeviceMatrix result(1, A.cols());


    cudaError_t err = cudaGetLastError();
    
    row_sum_op<rows_per_tile> <<<grid, block >>> (result.device_matrix, A.device_matrix, A.rows(), A.cols());
    
    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error("in procedure : matRowSumV2 : " + std::string(cudaGetErrorString(err)));
    }

    return result;

}




// OPREZ OVO JE TRIVIJALNA IMPLEMENTACIJA, TREBA REIMPLEMENTIRATI S PAMETNIJIM ALGORITMOM KOJI KORISTI COALESCED MEM ACCESS ZA UBRZANJE

__global__ void row_reduce_kernel(float* result, const float* matrix, size_t rows, size_t cols) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= cols)
        return;

    float sum = 0.0f;

    for (size_t row = 0; row < rows; ++row) {
        sum += matrix[row * cols + col];
    }

    result[col] = sum;
}

DeviceMatrix DeviceMatrix::matRowSum(const DeviceMatrix& A) {
    if (A.totalSize() == 0) {
        return A;
    }

    DeviceMatrix result(1, A.cols());

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((result.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (result.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    row_reduce_kernel << < dimGrid, dimBlock >> > (result.device_matrix, A.device_matrix, A.rows(), A.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}


// Identity matrix

__global__ void identity_kernel(float* matrix, size_t size) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size || col >= size) {
        return;
    }

    matrix[row * size + col] = (row == col) ? 1.0f : 0.0f;
}

DeviceMatrix DeviceMatrix::Identity(size_t size) {

    if (size == 0) {
        throw std::invalid_argument("Size must be greater than 0 for identity matrix.");
    }


    if (size < TILE_WIDTH || size < TILE_HEIGHT) {

        return DeviceMatrix(Matrix::Identity(size));
    }

    DeviceMatrix identity(size, size);

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);

    identity_kernel << <dimGrid, dimBlock >> > (identity.device_matrix, size);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return identity;
}


// Zero matrix

DeviceMatrix DeviceMatrix::Zero(size_t rows, size_t cols) {

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Rows and columns must be greater than 0 for zero matrix.");
    }

    DeviceMatrix zero(rows, cols);

    return zero;
}




__global__ void random_matrix_kernel(float* matrix, size_t rows, size_t cols, float min_val, float max_val, unsigned long seed)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;

    if (row >= rows || col >= cols) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float rand_val = curand_uniform(&state);

    matrix[idx] = min_val + rand_val * (max_val - min_val);

}


DeviceMatrix DeviceMatrix::Random(size_t rows, size_t cols, std::pair<float, float> range) {

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Rows and columns must be greater than 0 for random matrix.");
    }

    /*if (rows < TILE_HEIGHT || cols < TILE_WIDTH) {
        return DeviceMatrix(Matrix::Random(rows, cols, range));
    }*/

    DeviceMatrix random_matrix(rows, cols);

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_HEIGHT - 1) / TILE_HEIGHT);

    unsigned long seed = static_cast<unsigned long>(time(0));

    random_matrix_kernel <<<dimGrid, dimBlock >>> (random_matrix.device_matrix, rows, cols, range.first, range.second, seed);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return random_matrix;
}


// BROADCASTING OPERATIONS

__global__ void broadcast_add_kernel(float* result, const float* matrix, const float* vector, size_t rows, size_t cols) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;

    extern __shared__ float sh_vector[];

    if (row >= rows || col >= cols)
        return;


    if (threadIdx.x == 0) { // prvi stupac threadova u bloku ucitava cijeli stupac vektor u shared memoriju, prije je uvjet bilo col == 0, ali to je stvaralo problem jer samo prvi blok bi mogao ucitati vektor, a ostali blokovi bi bili prazni.
        sh_vector[row] = vector[row];
    }

    __syncthreads();

    float value = matrix[idx];
    float bias = sh_vector[row];

    result[idx] = value + bias;
}

DeviceMatrix DeviceMatrix::broadcastAdd(const DeviceMatrix& matrix, const DeviceMatrix& vector) {

    if (vector.cols() != 1 || vector.rows() != matrix.rows()) {
        throw std::runtime_error("Vector must be a single row with the same number of columns as the matrix.");
    }

    DeviceMatrix result(matrix.rows(), matrix.cols());

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((matrix.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (matrix.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    size_t shared_memory_size = vector.rows() * sizeof(float);

    broadcast_add_kernel << <dimGrid, dimBlock, shared_memory_size >> > (result.device_matrix, matrix.device_matrix, vector.device_matrix, matrix.rows(), matrix.cols());

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}


// ReLU-aktivacija

__global__ void ReLU_kernel(float* output, const float* input, size_t rows, size_t cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rows || col >= cols) {
        return;
    }

    int idx = row * cols + col;

    output[idx] = (input[idx] > 0) ? input[idx] : 0;
}


DeviceMatrix DeviceMatrix::ReLU(const DeviceMatrix& input) {

    if (input.totalSize() == 0) {
        throw std::runtime_error("ReLU forward error: Input matrix is empty");
    }

    DeviceMatrix output(input.rows(), input.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((input.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (input.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    ReLU_kernel << <gridSize, blockSize >> > (output.device_matrix, input.device_matrix, input.rows(), input.cols());

    cudaDeviceSynchronize();

    cudaError_t cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("ReLU forward error: " + std::string(cudaGetErrorString(cuda_error)));
    }

    return output;
}

__global__ void ReLU_gradient_kernel(float* output, const float* relu_output, size_t rows, size_t cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rows || col >= cols) {
        return;
    }

    int idx = row * cols + col;
    output[idx] = (relu_output[idx] > 0) ? 1.0f : 0.0f;
}

DeviceMatrix DeviceMatrix::ReLUGradient(const DeviceMatrix& relu_output) {

    if (relu_output.totalSize() == 0) {
        throw std::runtime_error("ReLU gradient error: Input matrix is empty");
    }

    DeviceMatrix output(relu_output.rows(), relu_output.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((relu_output.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (relu_output.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    ReLU_gradient_kernel << <gridSize, blockSize >> > (output.device_matrix, relu_output.device_matrix, relu_output.rows(), relu_output.cols());

    cudaDeviceSynchronize();

    cudaError_t cuda_error = cudaGetLastError();


    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("ReLU gradient error: " + std::string(cudaGetErrorString(cuda_error)));
    }

    return output;
}

// sigmoid activation

__global__ void sigmoid_kernel(float* output, const float* input, size_t rows, size_t cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}


DeviceMatrix DeviceMatrix::Sigmoid(const DeviceMatrix& input) {

    if (input.totalSize() == 0) {
        throw std::runtime_error("Sigmoid forward error: Input matrix is empty");
    }

    DeviceMatrix output(input.rows(), input.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((input.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (input.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    sigmoid_kernel << <gridSize, blockSize >> > (output.device_matrix, input.device_matrix, input.rows(), input.cols());

    cudaDeviceSynchronize();

    cudaError_t cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("Sigmoid forward error: " + std::string(cudaGetErrorString(cuda_error)));
    }

    return output;
}

__global__ void sigmoid_gradient_kernel(float* gradient, const float* sigmoid_output, size_t rows, size_t cols) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    float sigma = sigmoid_output[idx];

    gradient[idx] = sigma * (1.0f - sigma);
}

DeviceMatrix DeviceMatrix::SigmoidGradient(const DeviceMatrix& sigmoid_output) {

    if (sigmoid_output.totalSize() == 0) {
        throw std::runtime_error("Sigmoid gradient error: Input matrix or sigmoid value matrix is empty");
    }

    DeviceMatrix output(sigmoid_output.rows(), sigmoid_output.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((sigmoid_output.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (sigmoid_output.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    sigmoid_gradient_kernel << <gridSize, blockSize >> > (output.device_matrix, sigmoid_output.device_matrix, sigmoid_output.rows(), sigmoid_output.cols());

    cudaDeviceSynchronize();

    cudaError_t cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("Sigmoid gradient error: " + std::string(cudaGetErrorString(cuda_error)));
    }

    return output;
}

// mora se implementirati jos


__global__ void MSE_kernel(float* output, const float* target, size_t rows, size_t cols) {}

// POZOR, MSE TRENUTNO U TRIVIJALNOJ IMPLEMENTACIJI, SEMANTICKI ISPRAVAN, ALI NIKAKO OPTIMALAN PO PITANJU VREMENA

DeviceMatrix DeviceMatrix::MSE(const DeviceMatrix& output, const DeviceMatrix& target) {

    DeviceMatrix result = DeviceMatrix::matSub(output, target);

    result = DeviceMatrix::matElementWiseMul(result, result);

    result = DeviceMatrix::matRowSum(result);

    result = DeviceMatrix::matColSum(result);

    result = DeviceMatrix::matScale(result, 1.0f / (output.rows() * output.cols()));

    return result;
}


__global__ void MSE_gradient_kernel(float* gradient, const float* output, const float* target, size_t rows, size_t cols) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    gradient[idx] = 2.0f * (output[idx] - target[idx]) / (rows * cols);
}

// TRIVIJALNA IMPLEMENTACIJA, SEMANTICKI ISPRAVAN, ALI NIKAKO OPTIMALAN PO PITANJU VREMENA

DeviceMatrix DeviceMatrix::MSEGradient(const DeviceMatrix& output, const DeviceMatrix& target) {

    if (output.totalSize() == 0 || target.totalSize() == 0) {
        throw std::runtime_error("MSE gradient error: Output or target matrix is empty");
    }

    if (output.rows() != target.rows() || output.cols() != target.cols()) {
        throw std::runtime_error("MSE gradient error: Output and target matrices must have the same dimensions");
    }

    DeviceMatrix gradient(output.rows(), output.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((output.cols() + TILE_WIDTH - 1) / TILE_WIDTH, (output.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    MSE_gradient_kernel << <gridSize, blockSize >> > (gradient.device_matrix, output.device_matrix, target.device_matrix, output.rows(), output.cols());

    cudaDeviceSynchronize();

    cudaError_t cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("MSE gradient error: " + std::string(cudaGetErrorString(cuda_error)));
    }

    return gradient;
}

__global__ void BCE_gradient_kernel(float* gradient, const float* output, const float* target, size_t rows, size_t cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows || col >= cols)
        return;

    int idx = row * cols + col;

    float y_hat = output[idx];
    float y = target[idx];

    // epsilon da izbjegnemo log(0) i dijeljenje s 0
    const float eps = 1e-8f;
    gradient[idx] = (y_hat - y) / (max(y_hat * (1.0f - y_hat), eps));
}

DeviceMatrix DeviceMatrix::BCEGradient(const DeviceMatrix& output, const DeviceMatrix& target) {
    if (output.totalSize() == 0 || target.totalSize() == 0)
        throw std::runtime_error("BCE gradient: matrices empty");

    if (output.rows() != target.rows() || output.cols() != target.cols())
        throw std::runtime_error("BCE gradient: size mismatch");

    DeviceMatrix gradient(output.rows(), output.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((output.cols() + TILE_WIDTH - 1) / TILE_WIDTH,
        (output.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    BCE_gradient_kernel << <gridSize, blockSize >> > (
        gradient.device_matrix, output.device_matrix, target.device_matrix,
        output.rows(), output.cols()
        );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("BCE gradient kernel error: " + std::string(cudaGetErrorString(err)));

    return gradient;
}

__global__ void BCE_kernel(const float* output, const float* target, float* result, size_t rows, size_t cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    const float eps = 1e-8f; // stabilnost loga
    float y_hat = output[idx];
    float y = target[idx];

    result[idx] = -(y * logf(fmaxf(y_hat, eps)) + (1.0f - y) * logf(fmaxf(1.0f - y_hat, eps)));
}

DeviceMatrix DeviceMatrix::BCE(const DeviceMatrix& output, const DeviceMatrix& target) {
    if (output.totalSize() == 0 || target.totalSize() == 0)
        throw std::runtime_error("BCE: matrices empty");

    if (output.rows() != target.rows() || output.cols() != target.cols())
        throw std::runtime_error("BCE: size mismatch");


    DeviceMatrix bce_values(output.rows(), output.cols());

    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((output.cols() + TILE_WIDTH - 1) / TILE_WIDTH,
        (output.rows() + TILE_HEIGHT - 1) / TILE_HEIGHT);

    BCE_kernel << <gridSize, blockSize >> > (
        output.device_matrix,
        target.device_matrix,
        bce_values.device_matrix,
        output.rows(), output.cols()
        );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("BCE kernel error: " + std::string(cudaGetErrorString(err)));

 
    DeviceMatrix row_sum = DeviceMatrix::matRowSum(bce_values);
    DeviceMatrix total_sum = DeviceMatrix::matColSum(row_sum); 


    float scale = 1.0f / static_cast<float>(output.totalSize());
    DeviceMatrix mean_bce = DeviceMatrix::matScale(total_sum, scale);

    return mean_bce; 
}





};