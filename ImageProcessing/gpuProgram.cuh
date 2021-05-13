#pragma once
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudafeatures2d.hpp>
#include<opencv2/cudaarithm.hpp>

using namespace cv;

// function to process a bgr image on GPU
// returns the result
Mat processOnGPU(Mat bgrImage, bool useOpenCV);