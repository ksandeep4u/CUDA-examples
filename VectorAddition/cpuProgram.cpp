// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "cpuProgram.h"
#include<iostream>

using namespace std;

void add(int n, float* x, float* y) {
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
}

void cpuProgram(int n){
	cout << "CPU program!" << endl;
	int N = n;
	float* x = new float[N];
	float* y = new float[N];

	for (int i = 0; i < N; i++) {
		x[i] = 1;
		y[i] = 2;
	}

	add(N, x, y);

	// display result
	//for (int i = 0; i < N; i++)
	//	cout << y[i] << endl;

	//check for error
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	cout << "Max error = " << maxError << endl;

	delete x;
	delete y;
}
