// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#include "coordinateClassification.h"
#include "coordinatesDataset.h"
#include "../NeuralNetwork/neuralNetwork.cuh"
#include "../NeuralNetwork/linearLayer.cuh"
#include "../NeuralNetwork/reluLayer.cuh"
#include "../NeuralNetwork/sigmoidLayer.cuh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	// count the number of correct predictions
	int correctPredictions = 0;
	for (int i = 0; i < predictions.shape.x; i++) {
		// find the category with highest score
		float prediction = predictions[i] > 0.5 ? 1 : 0;

		// check if the highest scored category is the same as target
		if (prediction == targets[i])
			correctPredictions++;
	}

	// compute percentage of accuracy
	return static_cast<float> (correctPredictions) / predictions.shape.x;
}

void classifyCoordinates(bool useGPU) {
	// load coordinates dataset
	CoordinatesDataset dataset(100, 21);
	//dataset.print();

	// build network
	NeuralNetwork nn("BCE", useGPU); //BCE - Binary Cross Entropy (2 categories)
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30), useGPU));
	nn.addLayer(new ReluLayer("relu_1", useGPU));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1), useGPU));
	nn.addLayer(new SigmoidLayer("sigmoid_output", useGPU));

	Matrix predictions, targets;
	// train network
	for (int epoch = 0; epoch < 10001; epoch++) {
		//cout << "Epoch: " << epoch << endl;
		float cost = 0.0f;
		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			//cout << "Batch: " << batch << endl;
			predictions = nn.forward(dataset.getBatches().at(batch));
			targets = dataset.getTargets().at(batch);
			nn.backprop(predictions, targets);
			cost += nn.cost(predictions, targets);
		}

		if (epoch % 1000 == 0)
			cout << "Epoch: " << epoch << " Cost: " << cost / dataset.getNumOfBatches() << endl;
	}

	// test network: compute accuracy
	predictions = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	targets = dataset.getTargets().at(dataset.getNumOfBatches() - 1);
	float accuracy = computeAccuracy(predictions, targets);
	cout << "Accuracy = " << accuracy << endl;
}