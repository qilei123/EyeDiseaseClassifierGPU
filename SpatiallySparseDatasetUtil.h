#pragma once
#include "SpatiallySparseDataset.h"
#include "OpenCVPicture.h"
#include<string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

SpatiallySparseDataset KDRTrainSet(std::string dirName);
SpatiallySparseDataset KDRValidationSet(std::string dirName);
SpatiallySparseDataset KDRTestSet(std::string dirNameTest);
SpatiallySparseDataset KDRTestSet(std::string filename,int nClasses_);
SpatiallySparseDataset KDRTestSet(std::string filename,int nClasses_,cv::Mat &preSrc);

void distortImageColorDR(cv::Mat& mat, RNG& rng, float sigma0, float sigma1, float sigma2, float sigma3);
