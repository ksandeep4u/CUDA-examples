// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#include "coordinatesDataset.h"

CoordinatesDataset::CoordinatesDataset(size_t batch_size, size_t num_of_batches) :
	batch_size(batch_size), num_of_batches(num_of_batches)
{
	for (int i = 0; i < num_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 2)));
		targets.push_back(Matrix(Shape(batch_size, 1)));

		// allocate memory
		batches[i].allocateMemory();
		targets[i].allocateMemory();

		// fill data with random values
		for (int j = 0; j < batch_size; j++) {
			batches[i][j * 2 + 0] = static_cast<float> (rand()) / RAND_MAX - 0.5;
			batches[i][j * 2 + 1] = static_cast<float> (rand()) / RAND_MAX - 0.5;

			if (((batches[i][j * 2] < 0) && (batches[i][j * 2 + 1] < 0)) || ((batches[i][j * 2] > 0) && (batches[i][j * 2 + 1] > 0)))
				targets[i][j] = 1;
			else
				targets[i][j] = 0;
		}

		// copy to device
		//batches[i].copyHostToDevice();
		//targets[i].copyHostToDevice();
	}
}

int CoordinatesDataset::getNumOfBatches() {
	return num_of_batches;
}

vector<Matrix>& CoordinatesDataset::getBatches() {
	return batches;
}

vector<Matrix>& CoordinatesDataset::getTargets() {
	return targets;
}

void CoordinatesDataset::print() {
	for (int i = 0; i < num_of_batches; i++) {
		cout << "\nBatch = " << i << endl;
		for (int j = 0; j < batch_size; j++)
			cout << "(" << batches[i][j * 2] << "," << batches[i][j * 2 + 1] << ") --> " << targets[i][j] << endl;
	}
}