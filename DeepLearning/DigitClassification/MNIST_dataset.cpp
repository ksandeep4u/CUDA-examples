// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#include "MNIST_dataset.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

MNISTDataset::MNISTDataset(size_t batch_size, size_t num_of_batches) :
	batch_size(batch_size), num_of_batches(num_of_batches)
{
	ifstream myFile("DigitClassification/train.txt");
	if (!myFile.is_open()) {
		cout << "MNIST Dataset file not opened" << endl;
		return;
	}
	else
		cout << "Loading MNIST dataset (42,000 images) ..." << endl;

	for (int i = 0; i < num_of_batches; i++) {
		//cout << "Batch: " << i << endl;
		batches.push_back(Matrix(Shape(batch_size, 784)));
		targets.push_back(Matrix(Shape(batch_size, 10)));

		// allocate memory
		batches[i].allocateMemory();
		targets[i].allocateMemory();

		// fill data
		for (int j = 0; j < batch_size; j++) {
			string line;
			if (getline(myFile, line)) {
				stringstream ss(line);
				string word;
				vector<string> words;
				while (getline(ss, word, '\t'))
					words.push_back(word);
				int digit = stoi(words.at(0));

				for (int k = 0; k < 28 * 28; k++)
					batches[i][j * (28*28) + k] = stoi(words.at(k + 1)) / 255.0;

				for (int k = 0; k < 10; k++) {
					if (k == digit)
						targets[i][j * 10 + k] = 1;
					else
						targets[i][j * 10 + k] = 0;
				}
			}
			else
				cout << "End of file" << endl;
		}

		// copy to device
		//batches[i].copyHostToDevice();
		//targets[i].copyHostToDevice();
	}
}

int MNISTDataset::getNumOfBatches() {
	return num_of_batches;
}

vector<Matrix>& MNISTDataset::getBatches() {
	return batches;
}

vector<Matrix>& MNISTDataset::getTargets() {
	return targets;
}

void MNISTDataset::print() {
	for (int i = 0; i < num_of_batches; i++) {
		cout << "\nBatch = " << i << endl;
		for (int j = 0; j < batch_size; j++) {

			// generate gray image
			int iw = 28;
			int ih = 28;
			Mat grayImage(ih, iw, CV_8UC1, Scalar(0));
			for (int row = 0; row < ih; row++)
				for (int col = 0; col < iw; col++)
					grayImage.at<uchar>(row, col) = (unsigned char)(255 * batches[i][j * (iw*ih) + (row * iw + col)]);

			// print target
			cout << "Target = ";
			for (int k = 0; k < 10; k++)
				cout << targets[i][j * 10 + k];
			cout << endl;

			// display image
			imshow("Image", grayImage);
			waitKey(10);
		}
	}
}