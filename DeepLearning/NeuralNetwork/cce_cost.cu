// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "cce_cost.cuh"
#include <assert.h>
#include <math.h>
#include <algorithm>

__global__ void compute_CCECost(float* predictions, float* targets, int size, float* cost) {
	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = thIdx; index < size; index += stride){
		float partial_cost = targets[index] * logf(predictions[index]) + (1 - targets[index]) * logf(1 - predictions[index]);
		if (partial_cost != partial_cost) //Catch nan when target == prediction == 1 or target == prediction == 0
			partial_cost = 0.0f;
		atomicAdd(cost, -partial_cost/size);
	}
}

__global__ void compute_dCCECost(float* predictions, float* targets, float* dY, int size) {
	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = thIdx; index < size; index += stride)
		dY[index] = predictions[index] - targets[index];
}

float CCECost::cost(Matrix predictions, Matrix targets, bool useGPU) {
	assert(predictions.shape.x == targets.shape.x);
	float cost_value = 0;
	if (useGPU) {
		float* cost;
		cudaMallocManaged(&cost, sizeof(float));
		*cost = 0.0f;
		dim3 num_threads(256);
		dim3 num_blocks((predictions.shape.x + num_threads.x - 1) / num_threads.x);
		compute_CCECost << <num_blocks, num_threads >> > (predictions.data_device.get(), targets.data_device.get(), predictions.shape.x, cost);
		cudaDeviceSynchronize();
		cost_value = *cost;
		cudaFree(cost);
	}
	else {
		for (int idx = 0; idx < predictions.shape.x; idx++) {
			for (int k = 0; k < predictions.shape.y; k++) {
				int index = idx * predictions.shape.y + k;
				float partial_cost = targets[index] * logf(std::max(predictions[index], std::numeric_limits<float>::epsilon()));
				cost_value += -partial_cost;
			}
		}
		cost_value /= predictions.shape.x;
	}

	return cost_value;
}

Matrix CCECost::dCost(Matrix predictions, Matrix targets, Matrix dY, bool useGPU) {
	assert(predictions.shape.x == targets.shape.x);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((predictions.shape.x * predictions.shape.y + num_threads.x - 1) / num_threads.x);
		predictions.copyHostToDevice();
		targets.copyHostToDevice();
		dY.copyHostToDevice();
		compute_dCCECost << <num_blocks, num_threads >> > (predictions.data_device.get(), targets.data_device.get(), dY.data_device.get(), predictions.shape.x * predictions.shape.y);
		cudaDeviceSynchronize();
		dY.copyDeviceToHost();
	}
	else {
		for (int index = 0; index < predictions.shape.x * predictions.shape.y; index++)
			dY[index] = predictions[index] - targets[index];

		/*
		for (int idx = 0; idx < predictions.shape.x; idx++) {
			for (int k = 0; k < predictions.shape.y; k++) {
				int index = idx * predictions.shape.y + k;
				dY[index] = predictions[index] - targets[index];

				//dY[index] = -(1.0 / predictions.shape.x) * targets[index] / logf(std::max(predictions[index], std::numeric_limits<float>::epsilon()));
				//dY[index] = -(1.0 / predictions.shape.x) * targets[index] / std::max(predictions[index], std::numeric_limits<float>::epsilon());
			}
		}
		*/
	}

	return dY;
}
