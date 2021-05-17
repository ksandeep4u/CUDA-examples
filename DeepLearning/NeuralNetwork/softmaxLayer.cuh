// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "nnLayer.cuh"

// class holding a softmax layer: A = softmax(Z)
class SoftmaxLayer : public NNLayer {
private:
	Matrix Z; // input
	Matrix A; // output
	Matrix dZ;

public:
	SoftmaxLayer(std::string name, bool useGPU);
	~SoftmaxLayer();

	// forward propagation through the softmax layer
	Matrix& forward(Matrix Z);

	// backward propagation through the softmax layer
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};