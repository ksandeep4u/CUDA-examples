#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuProgram.cuh"

__global__ void gray_filter_kernel(unsigned char* bgrImage, unsigned char* gImage, int iw, int ih) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = gridDim.x * blockDim.x;
	int stride_y = gridDim.y * blockDim.y;

	for (int x = idx_x; x < iw; x += stride_x) {
		for (int y = idx_y; y < ih; y += stride_y) {
			int b = (int)bgrImage[3 * (y * iw + x) + 0];
			int g = (int)bgrImage[3 * (y * iw + x) + 1];
			int r = (int)bgrImage[3 * (y * iw + x) + 2];
			gImage[y * iw + x] = (unsigned char)(0.114076 * b + 0.585841 * g + 0.299083 * r);
		}
	}
}

__global__ void box_filter_kernel(unsigned char* gImage, unsigned char* bImage, int fSize, int iw, int ih) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = gridDim.x * blockDim.x;
	int stride_y = gridDim.y * blockDim.y;

	for (int x = idx_x; x < iw; x += stride_x) {
		for (int y = idx_y; y < ih; y += stride_y) {
			float sum = 0.0;
			int count = 0;
			for (int j = -fSize / 2; j <= fSize; j++) {
				for (int i = -fSize / 2; i <= fSize; i++) {
					if (((y + j) < ih) && ((y + j) >= 0) && ((x + i) < iw) && ((x + i) >= 0)) {
						sum += (float)gImage[(y + j) * iw + (x + i)];
						count += 1;
					}
				}
			}
			float avg = sum / (float)count;
			bImage[y * iw + x] = (unsigned char)avg;
		}
	}
}

__global__ void sobel_filter_kernel(unsigned char* gImage, unsigned char* sImage, int iw, int ih) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = gridDim.x * blockDim.x;
	int stride_y = gridDim.y * blockDim.y;

	for (int x = idx_x; x < iw; x += stride_x) {
		for (int y = idx_y; y < ih; y += stride_y) {
			if ((y < (ih - 1)) && (y > 0) && (x < (iw - 1)) && (x > 0)) {
				int gx = -gImage[(y - 1) * iw + (x - 1)] + gImage[(y - 1) * iw + (x + 1)] +
					-2 * gImage[y * iw + (x - 1)] + 2 * gImage[y * iw + (x + 1)] +
					-gImage[(y + 1) * iw + (x - 1)] + gImage[(y + 1) * iw + (x + 1)];
				int gy = -gImage[(y - 1) * iw + (x - 1)] - 2 * gImage[(y - 1) * iw + (x)] - gImage[(y - 1) * iw + (x + 1)] +
					gImage[(y + 1) * iw + (x - 1)] + 2 * gImage[(y + 1) * iw + (x)] + gImage[(y + 1) * iw + (x + 1)];

				sImage[y * iw + x] = (int)sqrt((float)(gx) * (float)(gx)+(float)(gy) * (float)(gy));
				//sImage[y * iw + x] = (int)(0.5 * gx + 0.5 * gy);
			}
		}
	}
}

__global__ void transpose_filter_kernel(unsigned char* gImage, unsigned char* tImage, int iw, int ih) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = gridDim.x * blockDim.x;
	int stride_y = gridDim.y * blockDim.y;

	for (int x = idx_x; x < iw; x += stride_x) {
		for (int y = idx_y; y < ih; y += stride_y) {
			tImage[x * ih + y] = gImage[y * iw + x];
		}
	}
}

Mat processOnGPU(Mat image_host, bool useOpenCV) {
	Mat grayImage_host;
	Mat blurredImage_host;
	Mat sobelImage_host;
	Mat transImage_host;

	int box_filter_size = 3; //filter size for box filter

	if (useOpenCV) { // using OpenCV::CUDA functions
		cuda::GpuMat image_device;
		image_device.upload(image_host);

		// gray conversion
		cuda::GpuMat grayImage_device;
		cuda::cvtColor(image_device, grayImage_device, CV_BGR2GRAY);
		grayImage_device.download(grayImage_host);

		// box filter
		cuda::GpuMat blurredImage_device;
		Ptr<cuda::Filter> bf = cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(box_filter_size, box_filter_size));
		bf->apply(grayImage_device, blurredImage_device);
		blurredImage_device.download(blurredImage_host);

		// sobel filter
		cuda::GpuMat sobelImage_dx, sobelImage_dy, sobelImage_device;
		Ptr<cuda::Filter> gx = cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 0);
		gx->apply(grayImage_device, sobelImage_dx);
		Ptr<cuda::Filter> gy = cuda::createSobelFilter(CV_8UC1, CV_8UC1, 0, 1);
		gy->apply(grayImage_device, sobelImage_dy);
		cuda::addWeighted(sobelImage_dx, 0.5, sobelImage_dy, 0.5, 0, sobelImage_device);
		sobelImage_device.download(sobelImage_host);

		// transpose
		cuda::GpuMat transImage_device;
		cuda::transpose(grayImage_device, transImage_device);
		transImage_device.download(transImage_host);
	}
	else { // using own kernel implementations
		int iw = image_host.size().width;
		int ih = image_host.size().height;
		dim3 blocks(ceil(iw / 32), ceil(ih / 32));
		dim3 threads(32, 32);

		unsigned char* image_device = NULL;
		cudaMalloc((void**)&image_device, iw * ih * 3);
		cudaMemcpy(image_device, image_host.data, iw * ih * 3, cudaMemcpyHostToDevice);

		// gray conversion
		grayImage_host = Mat(ih, iw, CV_8UC1, Scalar(0));
		unsigned char* grayImage_device = NULL;
		cudaMalloc((void**)&grayImage_device, iw * ih);
		cudaMemcpy(grayImage_device, grayImage_host.data, iw * ih, cudaMemcpyHostToDevice);
		gray_filter_kernel << <blocks, threads >> > (image_device, grayImage_device, iw, ih);
		cudaMemcpy(grayImage_host.data, grayImage_device, iw * ih, cudaMemcpyDeviceToHost);

		// box filter
		blurredImage_host = Mat(ih, iw, CV_8UC1, Scalar(100));
		unsigned char* blurredImage_device = NULL;
		cudaMalloc((void**)&blurredImage_device, iw * ih);
		cudaMemcpy(blurredImage_device, blurredImage_host.data, iw * ih, cudaMemcpyHostToDevice);
		box_filter_kernel << <blocks, threads >> > (grayImage_device, blurredImage_device, box_filter_size, iw, ih);
		cudaMemcpy(blurredImage_host.data, blurredImage_device, iw * ih, cudaMemcpyDeviceToHost);

		// sobel filter
		sobelImage_host = Mat(ih, iw, CV_8UC1, Scalar(0));
		unsigned char* sobelImage_device = NULL;
		cudaMalloc((void**)&sobelImage_device, iw * ih);
		cudaMemcpy(sobelImage_device, sobelImage_host.data, iw * ih, cudaMemcpyHostToDevice);
		sobel_filter_kernel << <blocks, threads >> > (grayImage_device, sobelImage_device, iw, ih);
		cudaMemcpy(sobelImage_host.data, sobelImage_device, iw * ih, cudaMemcpyDeviceToHost);

		// transpose
		transImage_host = Mat(iw, ih, CV_8UC1, Scalar(0));
		unsigned char* transImage_device = NULL;
		cudaMalloc((void**)&transImage_device, iw * ih);
		cudaMemcpy(transImage_device, transImage_host.data, iw * ih, cudaMemcpyHostToDevice);
		transpose_filter_kernel << <blocks, threads >> > (grayImage_device, transImage_device, iw, ih);
		cudaMemcpy(transImage_host.data, transImage_device, iw * ih, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		cudaFree(image_device);
		cudaFree(grayImage_device);
		cudaFree(blurredImage_device);
		cudaFree(sobelImage_device);
		cudaFree(transImage_device);
	}

	//Result
	Mat grayFrame_3c, blurredFrame_3c, sobelFrame_3c;
	cvtColor(grayImage_host, grayFrame_3c, CV_GRAY2BGR);
	cvtColor(blurredImage_host, blurredFrame_3c, CV_GRAY2BGR);
	cvtColor(sobelImage_host, sobelFrame_3c, CV_GRAY2BGR);
	Mat result, result1, result2;
	hconcat(image_host, grayFrame_3c, result1);
	hconcat(blurredFrame_3c, sobelFrame_3c, result2);
	vconcat(result1, result2, result);

	//imshow("Trans", transImage_host);
	//waitKey(10);

	return result;
}