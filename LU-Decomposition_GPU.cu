/*
The parallel algorithm of LU decomposition refers to page 12 of https://courses.engr.illinois.edu/cs554/fa2015/notes/06_lu.pdf
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>

__global__ void Paralled_kernel(double* As, int* minij_matrix, int k_iter, int N) {
	/*
		params: 
			As: the matrix stores the L and U component, which is updated iterately.
			k_iter: the k-th iteration of each kernel.
			N: the size of the matrix A.
	*/

	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int index = i * N + j; // As[index] = As[i][j]

	if (k_iter < minij_matrix[index]) {
		// This branch means the cell is not completed yet, thus it needs to be updated.

		As[index] -= As[i * N + k_iter] * As[k_iter * N + j];   // As[i][j] = As[i][j] - As[i][k_iter] * As[k_iter][j]
		__syncthreads();

		// Receive the broadcast from
		if (i > j) {
			As[index] /= As[j * N + j];
		}
	}

	__syncthreads();
}


__global__ void InitializeAs(double* As, int N) {

	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int index = i * N + j;

	if (j == 0 && i > 0) {
		As[index] /= As[0];
	}
	__syncthreads();
}


__global__ void Initialize_Minij_matrix(int* minij_matrix, int N) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int index = i * N + j; // As[index] = As[i][j]

	if (i <= j) {
		minij_matrix[index] = i;
	}
	else {
		minij_matrix[index] = j;
	}
	__syncthreads();
}


template<typename T>
void PrintMatrix(T* Mat, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << Mat[i*N +j] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "\n";
}



int main(int argc, char *argv[]){

	// The matrix to be be LU decomposed. OBS: You have to manually modify here.
	int N = 4;
	double A[] = { 2, 3, 2, 1.4, 1,3,2,-0.7, 3,-3, 4, 1, -3.2, 5.3, 4.5, 0.3 }; 
	double* h_As = A;
	double* d_As;
	cudaMalloc((void**)&d_As, sizeof(double) * N * N);
	cudaMemcpy(d_As, h_As, sizeof(double) * N * N, cudaMemcpyHostToDevice);

	// An assistive matrix to store min(i,j).
	int* d_minij_matrix;  
	cudaMalloc((void**)&d_minij_matrix, sizeof(int) * N * N);
	int* h_minij_matrix = (int*)malloc(sizeof(int) * N * N);
	
	
	// Define block and grid. OBS: You have to manually modify here.
	dim3 block(1, 1);
	dim3 grid(N, N);


	// ----------------- The algoithm starts here...
	// Step 0: 
	Initialize_Minij_matrix << <grid, block >> > (d_minij_matrix, N);
	cudaDeviceSynchronize();

	// Step 1:
	InitializeAs << <grid, block >> > (d_As, N);
	cudaDeviceSynchronize();

	// Step 2:
	for (int k = 0; k <= N - 1; k++) {
		Paralled_kernel << <grid, block >> > (d_As, d_minij_matrix, k, N);
	}

	cudaMemcpy(h_As, d_As, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
	PrintMatrix(h_As, N);


	// clear resources and exit
	cudaFree(d_As);
	cudaFree(d_minij_matrix);
	
	return 0;
}
