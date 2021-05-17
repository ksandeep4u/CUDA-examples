// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "../Utils/matrix.cuh"

// interface for the cost functions
class Cost {

public:
	// computes the cost
	virtual float cost(Matrix predictions, Matrix targets, bool useGPU) = 0;

	// computes the derivative of the cost
	virtual Matrix dCost(Matrix predictions, Matrix targets, Matrix dY, bool useGPU) = 0;
};