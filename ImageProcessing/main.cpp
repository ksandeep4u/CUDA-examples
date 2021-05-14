// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include<iostream>
#include "cpuProgram.h"
#include "gpuProgram.cuh"

using namespace std;

int main() {
	bool useGPU = true; //if true, runs on GPU, otherwise CPU
	bool useOpenCV = true; // if true, uses OpenCV functions, otherwise own implementations
	cout << "Press 'm' to switch between CPU and GPU" << endl;
	cout << "Press 'o' to switch between OpenCV and own kernels" << endl;

	// access camera
	VideoCapture camera(0);
	if (!camera.isOpened()) {
		cout << "Camera not opened" << endl;
		return -1;
	}

	while (1) {
		// read frame
		Mat frame;
		camera >> frame;

		double start = (double)cv::getTickCount(); // timer on

		// process frame
		Mat result;
		if (useGPU)
			result = processOnGPU(frame, useOpenCV);
		else
			result = processOnCPU(frame, useOpenCV);

		double stop = cv::getTickCount(); // timer off
		double timeElapsed = 1000 * (double)(stop - start) / cv::getTickFrequency();
		//cout << "Time elapsed = " << timeElapsed << "ms" << endl;

		// display result
		String str_time = String(to_string((int)timeElapsed) + "ms");
		String str_mode = useGPU? "GPU" : "CPU";
		String str_kernels = useOpenCV ? " (OpenCV)" : " (Own implementation)";
		putText(result, str_mode+str_kernels, Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));
		putText(result, str_time, Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));
		imshow("Output", result);
		char key = (char)waitKey(10);
		if (key == 'm') // press 'm' to switch between CPU and GPU
			useGPU = !useGPU;
		else if (key == 'o') // press 'o' to switch between OpenCV and own implementations
			useOpenCV = !useOpenCV;
		else if (key == 27)
			break;
	}

	camera.release();
	destroyAllWindows();

	return 0;
}