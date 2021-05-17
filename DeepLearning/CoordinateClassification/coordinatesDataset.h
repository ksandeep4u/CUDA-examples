// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "../Utils/matrix.cuh"
#include<vector>

// class holding the coordinates dataset
class CoordinatesDataset {
private:
	size_t batch_size; // number of samples per batch
	size_t num_of_batches; // number of batches
	vector<Matrix> batches;// input data
	vector<Matrix> targets;// groundtruth labels

public:
	CoordinatesDataset(size_t batch_size, size_t num_of_batches);
	
	// returns the number of batches
	int getNumOfBatches();

	// returns the input data
	vector<Matrix>& getBatches();

	// returns the groundtruth labels
	vector<Matrix>& getTargets();

	// prints the whole dataset
	void print();
};