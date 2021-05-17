// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "nnLayer.cuh"
#include "cost.cuh"
#include<vector>

// class holding the neural network
class NeuralNetwork {
private:
	vector<NNLayer*> layers; // pointers to network layers
	float learning_rate; // learning rate
	Cost* costFun; // cost function
	Matrix Y, dY; // intermediate outputs
	bool useGPU;

public:
	NeuralNetwork(string costFunName, bool useGPU, float learning_rate = 0.01);
	~NeuralNetwork();

	// adds a layer to the network
	void addLayer(NNLayer* layer);

	// returns network layers
	vector<NNLayer*> getLayers() const;

	// forward propagation through the network
	Matrix forward(Matrix X);

	// backward propagation through the network
	void backprop(Matrix predictions, Matrix targets);

	// computes the cost
	float cost(Matrix predictions, Matrix targets);
};