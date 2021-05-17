// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#include "DigitClassification.h"
#include "MNIST_dataset.h"
#include "../NeuralNetwork/neuralNetwork.cuh"
#include "../NeuralNetwork/linearLayer.cuh"
#include "../NeuralNetwork/reluLayer.cuh"
#include "../NeuralNetwork/softmaxLayer.cuh"
#include <iostream>

using namespace std;

float computeDCAccuracy(const Matrix& predictions, const Matrix& targets) {
	// count the number of correct predictions
	int correctPredictions = 0;
	for (int i = 0; i < predictions.shape.x; i++) {
		// find the category with highest score
		float max = 0.0;
		int maxIdx = -1;
		for (int k = 0; k < predictions.shape.y; k++) {
			int index = i * predictions.shape.y + k;
			if (predictions[index] > max) {
				max = predictions[index];
				maxIdx = index;
			}
		}

		// check if the highest scored category is the same as target
		if (targets[maxIdx] == 1)
			correctPredictions++;
	}

	// compute percentage of accuracy
	return static_cast<float> (correctPredictions) / predictions.shape.x;
}

void classifyDigits(bool useGPU) {
	// load MNIST dataset
	MNISTDataset dataset(2000, 21); //dataset has 42,000 images approximately
	//dataset.print();

	// build network
	NeuralNetwork nn("CCE", useGPU); //CCE - Categorical Cross Entropy (10 categories)
	nn.addLayer(new LinearLayer("linear_1", Shape(784, 128), useGPU));
	nn.addLayer(new ReluLayer("relu_1", useGPU));
	nn.addLayer(new LinearLayer("linear_2", Shape(128, 64), useGPU));
	nn.addLayer(new ReluLayer("relu_2", useGPU));
	nn.addLayer(new LinearLayer("linear_2", Shape(64, 10), useGPU));
	nn.addLayer(new SoftmaxLayer("softmax_output", useGPU));
	
	Matrix predictions, targets;
	// train network
	for (int epoch = 0; epoch < 1001; epoch++) {
		//cout << "Epoch: " << epoch << endl;
		float cost = 0.0f;
		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			//cout << "Batch: " << batch << endl;
			predictions = nn.forward(dataset.getBatches().at(batch));
			targets = dataset.getTargets().at(batch);
			nn.backprop(predictions, targets);
			cost += nn.cost(predictions, targets);
		}

		if (epoch % 100 == 0)
			cout << "Epoch: " << epoch << " Cost: " << cost / dataset.getNumOfBatches() << endl;
	}

	// test network: compute accuracy
	predictions = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	targets = dataset.getTargets().at(dataset.getNumOfBatches() - 1);
	float accuracy = computeDCAccuracy(predictions, targets);
	cout << "Accuracy = " << accuracy << endl;
}