// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "cost.cuh"

// class holding the Categorical Cross Entropy (CCE) cost
class CCECost : public Cost {

public:
	// computes the CCE cost
	float cost(Matrix predictions, Matrix targets, bool useGPU);

	// computes the derivative of the CCE cost
	Matrix dCost(Matrix predictions, Matrix targets, Matrix dY, bool useGPU);
};