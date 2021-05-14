// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include "cpuProgram.h"

void cvt_to_gray(Mat bgrImage, Mat gImage, int iw, int ih) {

	for (int y = 0; y < ih; y++) {
		for (int x = 0; x < iw; x++) {
			double b = (double) bgrImage.data[3 * (y * iw + x) + 0];
			double g = (double) bgrImage.data[3 * (y * iw + x) + 1];
			double r = (double) bgrImage.data[3 * (y * iw + x) + 2];
			gImage.data[y * iw + x] = (unsigned char)(0.114076 * b + 0.585841 * g + 0.299083 * r);
		}
	}
}

void box_filter(Mat gImage, Mat bImage, int fSize, int iw, int ih) {

	for (int y = 0; y < ih; y++) {
		for (int x = 0; x < iw; x++) {
			float sum = 0.0;
			int count = 0;
			for (int j = -fSize / 2; j <= fSize; j++) {
				for (int i = -fSize / 2; i <= fSize; i++) {
					if (((y + j) < ih) && ((y + j) >= 0) && ((x + i) < iw) && ((x + i) >= 0)) {
						sum += (float)gImage.data[(y + j) * iw + (x + i)];
						count += 1;
					}
				}
			}
			float avg = sum / (float)count;
			bImage.data[y * iw + x] = (unsigned char)avg;
		}
	}
	return;
}

void sobel_filter(Mat gImage, Mat sImage, int iw, int ih) {

	for (int y = 0; y < ih; y++) {
		for (int x = 0; x < iw; x++) {
			if ((y < (ih - 1)) && (y > 0) && (x < (iw - 1)) && (x > 0)) {
				int gx = -gImage.data[(y - 1) * iw + (x - 1)] + gImage.data[(y - 1) * iw + (x + 1)] +
					-2 * gImage.data[y * iw + (x - 1)] + 2 * gImage.data[y * iw + (x + 1)] +
					-gImage.data[(y + 1) * iw + (x - 1)] + gImage.data[(y + 1) * iw + (x + 1)];
				int gy = -gImage.data[(y - 1) * iw + (x - 1)] - 2 * gImage.data[(y - 1) * iw + (x)] - gImage.data[(y - 1) * iw + (x + 1)] +
					gImage.data[(y + 1) * iw + (x - 1)] + 2 * gImage.data[(y + 1) * iw + (x)] + gImage.data[(y + 1) * iw + (x + 1)];

				sImage.data[y * iw + x] = (int)sqrt((float)(gx) * (float)(gx)+(float)(gy) * (float)(gy));
			}
		}
	}
	return;
}

void transpose_fun(Mat gImage, Mat transImage, int iw, int ih) {

	for (int y = 0; y < ih; y++) {
		for (int x = 0; x < iw; x++) {
			if ((y < (ih - 1)) && (y > 0) && (x < (iw - 1)) && (x > 0)) {
				transImage.data[x * ih + y] = gImage.data[y * iw + x];
			}
		}
	}
	return;
}

Mat processOnCPU(Mat image, bool useOpenCV) {
	Mat grayImage;
	Mat blurredImage;
	Mat sobelImage;
	Mat transImage;

	int box_filter_size = 3; //filter size for box filter

	if (!useOpenCV) { // using own implementations
		int iw = image.size().width;
		int ih = image.size().height;

		// gray conversion
		grayImage = Mat(ih, iw, CV_8UC1, Scalar(0));
		cvt_to_gray(image, grayImage, iw, ih);

		// box filter
		blurredImage = Mat(ih, iw, CV_8UC1, Scalar(0));
		box_filter(grayImage, blurredImage, box_filter_size, iw, ih);

		// sobel filter
		sobelImage = Mat(ih, iw, CV_8UC1, Scalar(0));
		sobel_filter(grayImage, sobelImage, iw, ih);

		//transpose
		transImage = Mat(iw, ih, CV_8UC1, Scalar(0));
		transpose_fun(grayImage, transImage, iw, ih);
	}
	else { // using OpenCV functions
		// gray conversion
		cvtColor(image, grayImage, CV_BGR2GRAY);

		// box filter
		blur(grayImage, blurredImage, Size(box_filter_size, box_filter_size), Point(-1, -1));

		// sobel filter
		Mat gx, gy;
		Sobel(grayImage, gx, CV_16S, 1, 0);
		Sobel(grayImage, gy, CV_16S, 0, 1);
		Mat abs_gx, abs_gy;
		convertScaleAbs(gx, abs_gx);
		convertScaleAbs(gy, abs_gy);
		addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0, sobelImage);

		//transpose
		transpose(grayImage, transImage);
	}

	//Result
	Mat grayFrame_3c, blurredFrame_3c, sobelFrame_3c;
	cvtColor(grayImage, grayFrame_3c, CV_GRAY2BGR);
	cvtColor(blurredImage, blurredFrame_3c, CV_GRAY2BGR);
	cvtColor(sobelImage, sobelFrame_3c, CV_GRAY2BGR);
	Mat result, result1, result2;
	hconcat(image, grayFrame_3c, result1);
	hconcat(blurredFrame_3c, sobelFrame_3c, result2);
	vconcat(result1, result2, result);

	//imshow("Trans", transImage);
	//waitKey(10);

	return result;
}