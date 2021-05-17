// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#pragma once
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// structure holding the shape of the matrix
struct Shape {
	size_t x, y; //x is number of rows; y is the number of columns

	Shape(size_t x = 1, size_t y = 1) :
		x(x), y(y)
	{}
};

class Matrix {
private:
	bool device_allocated; // flag indicating if the memory is allocated on device
	bool host_allocated; // flag indicating if the memory is allocated on host

	// allocates memory on device
	void allocateDeviceMemory();

	// allocates memory on host
	void allocateHostMemory();

public:
	Shape shape; // shape of the matrix
	shared_ptr<float> data_device; //pointer to the device memory
	shared_ptr<float> data_host; // pointer to the host memory

	Matrix(size_t x = 1, size_t y = 1);
	Matrix(Shape shape);

	// allocate memory (host memory and device memory) for the matrix
	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	// copy matrix from host to device
	void copyHostToDevice();

	// copy matrix from device to host
	void copyDeviceToHost();

	// access a matrix element
	float& operator[](const int index);

	// access a matrix element
	const float& operator[](const int index) const;

	// print the matrix
	const void print();

	// transpose the matrix
	Matrix transpose();

	// multiply with matrix X
	Matrix operator*(Matrix);
};