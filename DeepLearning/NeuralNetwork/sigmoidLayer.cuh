// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "nnLayer.cuh"

// class holding a sigmoid layer: A = sigmoid(Z)
class SigmoidLayer : public NNLayer {
private:
	Matrix Z; // input
	Matrix A; // output
	Matrix dZ;

public:
	SigmoidLayer(std::string name, bool useGPU);
	~SigmoidLayer();

	// forward propagation through the sigmoid layer
	Matrix& forward(Matrix Z);

	// backward propagation through the sigmoid layer
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};