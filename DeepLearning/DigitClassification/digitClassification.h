// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "../Utils/matrix.cuh"

// computes the accuracy based on predictions and targets (groundtruth)
float computeAccuracy(const Matrix& predictions, const Matrix& targets);

// builds, trains and tests the digit classification model
void classifyDigits(bool useGPU);
