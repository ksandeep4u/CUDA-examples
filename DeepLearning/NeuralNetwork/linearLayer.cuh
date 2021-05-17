// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "nnLayer.cuh"

// class holding a linear layer: Z = AW+b
class LinearLayer : public NNLayer {
private:
	Matrix W; // weight
	Matrix b; // bias
	Matrix A; // input
	Matrix dA;
	Matrix Z; // output
	const float weights_init_threshold = 0.01;

	// initialise bias
	void initBiasWithZeros();

	// initialise weights
	void initWeightsRandomly();

	// update bias
	void updateBias(Matrix& dZ, float learning_rate);

	// update weights
	void updateWeights(Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape shape, bool useGPU);
	~LinearLayer();
	
	// forward propagation through the linear layer
	Matrix& forward(Matrix A);

	// backward propagation through the linear layer
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	//int getXDim() const;
	//int getYDim() const;
	//Matrix getWeightMatrix() const;
	//Matrix getBiasVector() const;
};