// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "nnLayer.cuh"

// class holding a relu layer: A = relu(Z)
class ReluLayer : public NNLayer {
private:
	Matrix Z; // input
	Matrix A; // output
	Matrix dZ;

public:
	ReluLayer(std::string name, bool useGPU);
	~ReluLayer();

	// forward propagation through the relu layer
	Matrix& forward(Matrix Z);

	// backward propagation through the relu layer
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};