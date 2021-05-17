// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "softmaxLayer.cuh"

__global__ void softmaxLayerForward(float* Z, float* A, int Zx, int Zy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	// TODO implement kernel

}

__global__ void softmaxLayerBackProp(float* Z, float* dA, float* dZ, int Zx, int Zy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	// TODO implement kernel
}


SoftmaxLayer::SoftmaxLayer(std::string name, bool useGPU){
	this->name = name;
	this->useGPU = useGPU;
}

SoftmaxLayer::~SoftmaxLayer() {}

Matrix& SoftmaxLayer::forward(Matrix Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((Z.shape.x * Z.shape.y + num_threads.x - 1) / num_threads.x);
		softmaxLayerForward << <num_blocks, num_threads >> > (Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
		cudaDeviceSynchronize();
		A.copyDeviceToHost();
	}
	else {
		for (int idx = 0; idx < Z.shape.x; idx++) {
			double max_val = -INFINITY;
			for (int k = 0; k < Z.shape.y; k++) {
				int index = idx * Z.shape.y + k;
				if (Z[index] > max_val)
					max_val = Z[index];
			}

			double sum = 0.0;
			for (int k = 0; k < Z.shape.y; k++) {
				int index = idx * Z.shape.y + k;
				sum += exp(Z[index] - max_val);
			}

			for (int k = 0; k < Z.shape.y; k++) {
				int index = idx * Z.shape.y + k;
				A[index] = exp(Z[index] - max_val) / sum;
			}
		}
	}

	return A;
}

Matrix& SoftmaxLayer::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((Z.shape.x * Z.shape.y + num_threads.x - 1) / num_threads.x);
		softmaxLayerBackProp << <num_blocks, num_threads >> > (Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);
		cudaDeviceSynchronize();
		dZ.copyDeviceToHost();
	}
	else {
		for (int idx = 0; idx < Z.shape.x; idx++) {
			for (int k = 0; k < Z.shape.y; k++) {
				int index = idx * Z.shape.y + k;
				dZ[index] = dA[index];
				/*
				for (int k_ = 0; k_ < Z.shape.y; k_++) {
					int index_ = k_ * Z.shape.x + idx; //idx * 10 + k
					double delta = 1.0;
					if (k == k_)
						delta = 1.0;
					else
						delta = 0.0;
					dZ[index] += dA[index_] * A[index] * (delta - A[index_]);
				}
				*/
			}
		}
	}

	return dZ;
}
