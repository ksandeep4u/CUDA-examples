// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "bce_cost.cuh"
#include <assert.h>
#include <math.h>
#include <algorithm>

__global__ void compute_BCECost(float* predictions, float* targets, int size, float* cost) {
	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = thIdx; index < size; index += stride){
		float partial_cost = targets[index] * logf(predictions[index]) + (1 - targets[index]) * logf(1 - predictions[index]);
		if (partial_cost != partial_cost) //Catch nan when target == prediction == 1 or target == prediction == 0
			partial_cost = 0.0f;
		atomicAdd(cost, -partial_cost/size);
	}
}

__global__ void compute_dBCECost(float* predictions, float* targets, float* dY, int size) {
	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = thIdx; index < size; index += stride)
		dY[index] = predictions[index] - targets[index];
	
	/*
	for (int index = thIdx; index < size; index += stride){
		dY[thIdx] = -1.0 * (targets[thIdx] / predictions[thIdx] - (1 - targets[thIdx]) / (1 - predictions[thIdx]));
		if (dY[thIdx] != dY[thIdx]) //Catch nan when target == prediction == 1 or target == prediction == 0
			if (targets[thIdx] == 0)
				dY[thIdx] = 1.0f;
			else if (targets[thIdx] == 1)
				dY[thIdx] = -1.0f;
	}
	*/
}

float BCECost::cost(Matrix predictions, Matrix targets, bool useGPU) {
	assert(predictions.shape.x == targets.shape.x);
	float cost_value = 0;
	if (useGPU){
		float* cost;
		cudaMallocManaged(&cost, sizeof(float));
		*cost = 0.0f;
		dim3 num_threads(256);
		dim3 num_blocks((predictions.shape.x+num_threads.x-1)/num_threads.x);
		compute_BCECost << <num_blocks, num_threads >> > (predictions.data_device.get(), targets.data_device.get(), predictions.shape.x, cost);
		cudaDeviceSynchronize();
		cost_value = *cost;
		cudaFree(cost);
	}
	else {
		for (int idx = 0; idx < predictions.shape.x; idx++) {
			float partial_cost = targets[idx] * logf(std::max(predictions[idx], std::numeric_limits<float>::epsilon())) + (1 - targets[idx]) * logf(std::max(1 - predictions[idx], std::numeric_limits<float>::epsilon()));
			//float partial_cost = targets[idx] * logf(predictions[idx]) + (1 - targets[idx]) * logf(1 - predictions[idx]);
			if (partial_cost != partial_cost) { //Catch nan when target == prediction == 1 or target == prediction == 0
				partial_cost = 0.0f;
				cout << "P Cost NaN" << endl;
			}
			cost_value += -partial_cost;
			//cost_value += -partial_cost / (float)predictions.shape.x;
		}
		cost_value /= predictions.shape.x;
	}

	return cost_value;
}

Matrix BCECost::dCost(Matrix predictions, Matrix targets, Matrix dY, bool useGPU) {
	assert(predictions.shape.x == targets.shape.x);
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((predictions.shape.x + num_threads.x - 1) / num_threads.x);
		predictions.copyHostToDevice();
		targets.copyHostToDevice();
		dY.copyHostToDevice();
		compute_dBCECost << <num_blocks, num_threads >> > (predictions.data_device.get(), targets.data_device.get(), dY.data_device.get(), predictions.shape.x);
		cudaDeviceSynchronize();
		dY.copyDeviceToHost();
	}
	else {
		for (int idx = 0; idx < predictions.shape.x; idx++)
			dY[idx] = predictions[idx] - targets[idx];

		/*
		for (int idx = 0; idx < predictions.shape.x; idx++) {
			dY[idx] = -1.0 * (targets[idx] / predictions[idx] - (1 - targets[idx]) / (1 - predictions[idx]));
			if (dY[idx] != dY[idx]) { //Catch nan when target == prediction == 1 or target == prediction == 0
				if (targets[idx] == 0)
					dY[idx] = 1.0f;
				else if (targets[idx] == 1)
					dY[idx] = -1.0f;
			}
		}
		*/
	}

	return dY;
}
