// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "../Utils/matrix.cuh"

// interface holding a network layer
class NNLayer {
protected:
	std::string name; // layer name
	bool useGPU;

public:
	virtual ~NNLayer() = 0;

	// forward propagation through the layer
	virtual Matrix& forward(Matrix input) = 0;

	// backward propagation through the layer
	virtual Matrix& backprop(Matrix& errorDerivative, float learning_rate) = 0;

	// return the layer name
	std::string getName() {
		return this->name;
	}
};

inline NNLayer::~NNLayer(){}