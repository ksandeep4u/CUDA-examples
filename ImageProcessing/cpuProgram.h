// Created By: Sandeep Katragadda
// https://github.com/ksandeep4u/CUDA-examples

#pragma once
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace cv;

// function converts bgr image to gray
void cvt_to_gray(Mat bgrImage, Mat grayOutput, int width, int height);

// function runs box filter on a gray image
void box_filter(Mat grayImage, Mat output, int filter_size, int width, int height);

// function runs sobel filter on a gray image
void sobel_filter(Mat grayImage, Mat output, int width, int height);

// function transposes a gray image
void transpose_fun(Mat grayImage, Mat output, int width, int height);

// function to process a bgr image on CPU
// returns the result
Mat processOnCPU(Mat bgrImage, bool useOpenCV);