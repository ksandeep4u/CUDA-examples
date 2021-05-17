// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#include <iostream>
#include "CoordinateClassification/coordinateClassification.h"
#include "DigitClassification/digitClassification.h"

using namespace std;

int main() {
	cout << "Building and training my first neural network using CUDA" << endl;
	bool useGPU = false;

	classifyCoordinates(useGPU);
	classifyDigits(useGPU);

	return 0;
}

