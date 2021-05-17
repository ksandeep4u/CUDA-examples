// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "sigmoidLayer.cuh"

__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidLayerForward(float* Z, float* A, int Zx, int Zy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int index = 0; index < Zx * Zy; index += stride)
		A[index] = sigmoid(Z[index]);
}

__global__ void sigmoidLayerBackProp(float* Z, float* dA, float* dZ, int Zx, int Zy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int index = 0; index < Zx * Zy; index += stride)
		dZ[index] = dA[index];
}

SigmoidLayer::SigmoidLayer(std::string name, bool useGPU){
	this->name = name;
	this->useGPU = useGPU;
}

SigmoidLayer::~SigmoidLayer() {}

Matrix& SigmoidLayer::forward(Matrix Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((Z.shape.x * Z.shape.y + num_threads.x - 1) / num_threads.x);
		sigmoidLayerForward << <num_blocks, num_threads >> > (Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
		cudaDeviceSynchronize();
		A.copyDeviceToHost();
	}
	else
		for (int index = 0; index < Z.shape.x * Z.shape.y; index++) {
			A[index] = 1.0f / (1.0f + exp(-Z[index]));
	}

	return A;
}

Matrix& SigmoidLayer::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((Z.shape.x * Z.shape.y + num_threads.x - 1) / num_threads.x);
		sigmoidLayerBackProp << <num_blocks, num_threads >> > (Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);
		cudaDeviceSynchronize();
		dZ.copyDeviceToHost();
	}
	else {
		for (int index = 0; index < Z.shape.x * Z.shape.y; index++) {
			dZ[index] = dA[index];
			/*
			dZ[index] = dA[index] * (1.0f / (1 + exp(-Z[index]))) * (1 - (1.0f / (1 + exp(-Z[index]))));
			*/
		}
	}

	return dZ;
}
