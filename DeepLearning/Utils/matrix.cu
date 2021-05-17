// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples
// Modified version of https://github.com/pwlnk/cuda-neural-network

#include "matrix.cuh"
#include <assert.h>

Matrix::Matrix(size_t x, size_t y) :
	shape(x, y), data_device(nullptr), data_host(nullptr), device_allocated(false), host_allocated(false)
{}

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{}

void Matrix::allocateDeviceMemory() {
	if (!device_allocated) {
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		data_device = shared_ptr<float>(device_memory, [&](float* ptr) {cudaFree(ptr); });
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		float* host_memory = new float[shape.x * shape.y];
		data_host = shared_ptr<float>(host_memory, [&](float* ptr) {delete[] ptr; });
		host_allocated = true;
	}
}

void Matrix::allocateMemory() {
	allocateDeviceMemory();
	allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated)
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y, cudaMemcpyDeviceToHost);
	else
		cout << "Memory not allocated" << endl;
}

void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated)
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y, cudaMemcpyHostToDevice);
	else
		cout << "Memory not allocated" << endl;
}

float& Matrix::operator[](const int index) {
	return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
	return data_host.get()[index];
}

const void Matrix::print() {
	cout << "Shape: (" << shape.x << ", " << shape.y << ")" << endl;
	cout << "Data: " << endl;
	for (int row = 0; row < shape.x; row++) {
		for (int col = 0; col < shape.y; col++) {
			int index = row * shape.y + col;
			cout << (*this)[index] << "\t";
		}
		cout << endl;
	}
}

Matrix Matrix::transpose() {
	Matrix result = Matrix(shape.y, shape.x);
	result.allocateMemory();
	for (int row = 0; row < shape.x; row++) {
		for (int col = 0; col < shape.y; col++) {
			int index = row * shape.y + col;
			int index_res = col * shape.x + row;
			result[index_res] = (*this)[index];
		}
	}
	return result;
}

Matrix Matrix::operator*(Matrix X){
	assert(shape.y == X.shape.x);
	Matrix result = Matrix(shape.x, X.shape.y);
	result.allocateMemory();
	for (int row = 0; row < result.shape.x; row++) {
		for (int col = 0; col < result.shape.y; col++) {
			float Z_tmp = 0.0f;
			for (int i = 0; i < X.shape.x; i++)
				Z_tmp += (*this)[row * shape.y + i] * X[i * X.shape.y + col];
			result[row * result.shape.y + col] = Z_tmp;
		}
	}
	return result;
}