// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "cost.cuh"

// class holding the Binary Cross Entropy (BCE) cost
class BCECost : public Cost {

public:
	// computes the BCE cost
	float cost(Matrix predictions, Matrix targets, bool useGPU);

	// computes the derivative of the BCE cost
	Matrix dCost(Matrix predictions, Matrix targets, Matrix dY, bool useGPU);
};