// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#include "neuralNetwork.cuh"
#include "bce_cost.cuh"
#include "cce_cost.cuh"

NeuralNetwork:: NeuralNetwork(string costFunName, bool useGPU, float learning_rate):
	learning_rate(learning_rate), useGPU(useGPU)
{
	if (costFunName == "BCE")  //BCE - Binary Cross Entropy
		costFun = new BCECost();
	else if (costFunName == "CCE")  //CCE - Categorical Cross Entropy
		costFun = new CCECost();
	else
		cout << "Inavlid cost" << endl;
}

NeuralNetwork::~NeuralNetwork(){
	for (auto layer : layers)
		delete layer;
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	layers.push_back(layer);
}

vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;
	// forward propagation through each layer
	for (auto layer : layers)
		Z = layer->forward(Z);
	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix targets) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);

	// compute derivate of the cost
	Matrix error = costFun->dCost(predictions, targets, dY, useGPU);

	// backward propagation through each layer
	for (auto it = layers.rbegin(); it != layers.rend(); it++)
		error = (*it)->backprop(error, learning_rate);
}

float NeuralNetwork::cost(Matrix predictions, Matrix targets) {
	// compute the cost
	return costFun->cost(predictions, targets, useGPU);
}
