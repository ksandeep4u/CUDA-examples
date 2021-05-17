// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "linearLayer.cuh"
#include <assert.h>
#include <random>

__global__ void linearLayerForward(float* W, float* A, float* b, float* Z, int Wx, int Wy, int Ax, int Ay) {
	int rIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int cIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int stride_r = gridDim.x * blockDim.x;
	int stride_c = gridDim.y * blockDim.y;

	int Zx = Ax;
	int Zy = Wy;

	for (int row = rIdx; row < Zx; row+=stride_r) {
		for (int col = cIdx; col < Zy; col+=stride_c) {
			float Z_tmp = 0.0f;
			for (int i = 0; i < Wx; i++)
				Z_tmp += A[row * Ay + i] * W[i * Wy + col];
			Z[row * Zy + col] = Z_tmp + b[col];
		}
	}
}

__global__ void linearLayerBackProp(float* W, float* dZ, float* dA, int Wx, int Wy, int dZx, int dZy) {
	int rIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int cIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int stride_r = gridDim.x * blockDim.x;
	int stride_c = gridDim.y * blockDim.y;

	int dAx = dZx;
	int dAy = Wx;
	for (int row = rIdx; row < dAx; row += stride_r) {
		for (int col = cIdx; col < dAy; col += stride_c) {
			float dA_tmp = 0.0f;
			for (int i = 0; i < Wy; i++)
				dA_tmp += dZ[row * dZy + i] * W[col * Wy + i];
			dA[row * dAy + col] = dA_tmp;
		}
	}
}

__global__ void linearLayerUpdateBias(float* dZ, float* b, int dZx, int dZy, int bx, float learning_rate) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int index = idx; index < dZx*dZy; index += stride) {
		int col = index % dZy;
		int row = index / dZy;
		atomicAdd(&b[col], -learning_rate * (dZ[row * dZy + col] / dZx));
	}
}

__global__ void linearLayerUpdateWeights(float* dZ, float* A, float* W, int dZx, int dZy, int Ax, int Ay, float learning_rate) {
	int rIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int cIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int stride_r = gridDim.x * blockDim.x;
	int stride_c = gridDim.y * blockDim.y;

	int Wx = Ay;
	int Wy = dZy;
	for (int row = rIdx; row < Wx; row += stride_r) {
		for (int col = cIdx; col < Wy; col += stride_c) {
			float dW_tmp = 0.0f;
			for (int i = 0; i < dZx; i++)
				dW_tmp += A[i * Ay + row] * dZ[i * dZy + col];
			W[row * Wy + col] -= (float)(learning_rate * (dW_tmp / Ax));
		}
	}
}

LinearLayer::LinearLayer(std::string name, Shape shape, bool useGPU):
	W(shape), b(shape.y,1)
{
	this->name = name;
	this->useGPU = useGPU;
	b.allocateMemory();
	W.allocateMemory();
	initBiasWithZeros();
	initWeightsRandomly();
}

LinearLayer::~LinearLayer(){}

void LinearLayer::initBiasWithZeros() {
	for (int i = 0; i < b.shape.x; i++)
		b[i] = 0.0f;
	//b.copyHostToDevice();
}

void LinearLayer::initWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution;
	for (int i = 0; i < W.shape.x; i++)
		for (int j = 0; j < W.shape.y; j++)
			W[i * W.shape.y + j] = normal_distribution(generator) * weights_init_threshold;
	//W.copyHostToDevice();
}

// Z = AW+b
Matrix& LinearLayer::forward(Matrix A) {
	assert(W.shape.x == A.shape.y);
	this->A = A;
	int Zx = A.shape.x;
	int Zy = W.shape.y;
	Z.allocateMemoryIfNotAllocated(Shape(A.shape.x, W.shape.y));
	if (useGPU) {
		dim3 num_threads(256, 256);
		dim3 num_blocks((Zx + num_threads.x - 1) / num_threads.x, (Zy + num_threads.y - 1) / num_threads.y);
		W.copyHostToDevice();
		A.copyHostToDevice();
		b.copyHostToDevice();
		linearLayerForward << <num_blocks, num_threads >> > (W.data_device.get(), A.data_device.get(), b.data_device.get(), Z.data_device.get(), W.shape.x, W.shape.y, A.shape.x, A.shape.y);
		cudaDeviceSynchronize();
		Z.copyDeviceToHost();
	}
	else {
		Z = A * W;
		for (int row = 0; row < Zx; row++)
			for (int col = 0; col < Zy; col++)
				Z[row * Zy + col] += b[col];
	}

	return Z;
}

// dA = dZ*transpose(W)
Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
	dA.allocateMemoryIfNotAllocated(A.shape);
	assert(dA.shape.x == dZ.shape.x);
	assert(dA.shape.y == W.shape.x);
	if (useGPU) {
		dim3 num_threads(256, 256);
		dim3 num_blocks((A.shape.x + num_threads.x - 1) / num_threads.x, (A.shape.y + num_threads.y - 1) / num_threads.y);
		dZ.copyHostToDevice();
		linearLayerBackProp << <num_blocks, num_threads >> > (W.data_device.get(), dZ.data_device.get(), dA.data_device.get(), W.shape.x, W.shape.y, dZ.shape.x, dZ.shape.y);
		cudaDeviceSynchronize();
		dA.copyDeviceToHost();
	}
	else
		dA = dZ * W.transpose();

	updateBias(dZ, learning_rate);
	updateWeights(dZ, learning_rate);

	return dA;
}

void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {
	if (useGPU) {
		dim3 num_threads(256);
		dim3 num_blocks((dZ.shape.x + num_threads.x - 1) / num_threads.x, (dZ.shape.y + num_threads.y - 1) / num_threads.y);
		dZ.copyHostToDevice();
		linearLayerUpdateBias << <num_blocks, num_threads >> > (dZ.data_device.get(), b.data_device.get(), dZ.shape.x, dZ.shape.y, b.shape.x, learning_rate);
		cudaDeviceSynchronize();
		b.copyDeviceToHost();
	}
	else {
		for (int index = 0; index < dZ.shape.x * dZ.shape.y; index++) {
			int col = index % dZ.shape.y;
			int row = index / dZ.shape.y;
			b[col] -= learning_rate * (dZ[row * dZ.shape.y + col] / dZ.shape.x);
		}
	}
}

void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {
	assert(W.shape.x == A.shape.y);
	assert(W.shape.y == dZ.shape.y);

	if (useGPU) {
		dim3 num_threads(256, 256);
		dim3 num_blocks((W.shape.x + num_threads.x - 1) / num_threads.x, (W.shape.y + num_threads.y - 1) / num_threads.y);
		dZ.copyHostToDevice();
		linearLayerUpdateWeights << <num_blocks, num_threads >> > (dZ.data_device.get(), A.data_device.get(), W.data_device.get(), dZ.shape.x, dZ.shape.y, A.shape.x, A.shape.y, learning_rate);
		cudaDeviceSynchronize();
		W.copyDeviceToHost();
	}
	else {
		Matrix dW = A.transpose() * dZ;
		for (int row = 0; row < W.shape.x; row++) {
			for (int col = 0; col < W.shape.y; col++) {
				W[row * W.shape.y + col] -= (float)(learning_rate * (dW[row * dW.shape.y + col] / A.shape.x));
			}
		}
	}
}
