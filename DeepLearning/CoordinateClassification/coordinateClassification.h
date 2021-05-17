// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include "../Utils/matrix.cuh"

// computes the accuracy based on predictions and targets (groundtruth)
float computeAccuracy(const Matrix& predictions, const Matrix& targets);

// builds, trains and tests the coordinate classification model
void classifyCoordinates(bool useGPU);