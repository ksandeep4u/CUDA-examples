// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuProgram.cuh"
#include<iostream>
#include<string>

using namespace std;

// CUDA kernel
__global__ void add_kernel(int n, float* x, float* y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < n; i += stride)
		y[i] = x[i] + y[i];
}

void gpuProgram(int n) {
	cout << "GPU program!" << endl;
	int v = 0;
	cudaRuntimeGetVersion(&v);
	string version = to_string(v / 1000) + "." + to_string(v % 1000);
	cout << "CUDA run time version: " << version << endl;
	cudaDriverGetVersion(&v);
	version = to_string(v / 1000) + "." + to_string(v % 1000);
	cout << "CUDA driver version: " << version << endl;

	int N = n;

	// initialization
	float* x;
	float* y;

	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1;
		y[i] = 2;
	}

	// call to kernel
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add_kernel << <numBlocks, blockSize >> > (N, x, y);
	cudaDeviceSynchronize();

	// display result
	//for (int i = 0; i < N; i++)
	//	cout << y[i] << endl;

	//check for error
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	cout << "Max error = " << maxError << endl;

	cudaFree(x);
	cudaFree(y);
}
