// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "reluLayer.cuh"

__global__ void reluLayerForward(float* Z, float* A, int size) {
	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = thIdx; index < size; index += stride)
		A[index] = fmaxf(Z[index], 0.0f);
}

__global__ void reluLayerBackProp(float* Z, float* dA, float* dZ, int size) {
	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = thIdx; index < size; index += stride){
		if (Z[index] > 0)
			dZ[index] = dA[index];
		else
			dZ[index] = 0;
	}
}


ReluLayer::ReluLayer(std::string name, bool useGPU){
	this->name = name;
	this->useGPU = useGPU;
}

ReluLayer::~ReluLayer(){}

Matrix& ReluLayer::forward(Matrix Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((Z.shape.x * Z.shape.y + num_threads.x - 1) / num_threads.x);
		Z.copyHostToDevice();
		reluLayerForward << <num_blocks, num_threads >> > (Z.data_device.get(), A.data_device.get(), Z.shape.x * Z.shape.y);
		cudaDeviceSynchronize();
		A.copyDeviceToHost();
	}
	else {
		for (int index = 0; index < Z.shape.x * Z.shape.y; index++)
			A[index] = fmaxf(Z[index], 0.0f);
	}

	return A;
}

Matrix& ReluLayer::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((Z.shape.x * Z.shape.y + num_threads.x - 1) / num_threads.x);
		Z.copyHostToDevice();
		dA.copyHostToDevice();
		reluLayerBackProp << <num_blocks, num_threads >> > (Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x * Z.shape.y);
		cudaDeviceSynchronize();
		dZ.copyDeviceToHost();
	}
	else {
		for (int index = 0; index < Z.shape.x * Z.shape.y; index++) {
			if (Z[index] > 0)
				dZ[index] = dA[index];
			else
				dZ[index] = 0;
		}
	}

	return dZ;
}
