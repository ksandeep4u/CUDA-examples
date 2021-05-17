// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "../Utils/matrix.cuh"
#include<vector>

class MNISTDataset {
private:
	size_t batch_size;
	size_t num_of_batches;
	vector<Matrix> batches;
	vector<Matrix> targets;

public:
	MNISTDataset(size_t batch_size, size_t num_of_batches);
	int getNumOfBatches();
	vector<Matrix>& getBatches();
	vector<Matrix>& getTargets();
	void print();
};