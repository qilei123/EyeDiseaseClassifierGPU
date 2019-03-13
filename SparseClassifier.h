#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetUtil.h"
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <stdio.h>
#include <vector>
#include "Imagenet.h"

#define PREPROCESS 0 //图像增强
#define PREPROCESS_CP 1 //图像增强+图像裁剪
#define PREPROCESS_CO 2 //图像裁剪

//uerinterface class
class EyeDiseaseClassifier{
	private:
		Imagenet *cnn;
		int nClasses = 3;
		int nFeatures = 3;
		int cudaDevice = -1;
		int batchSize=1;
		int nTime=5;
		int processFlag = 1;
		std::string classifierName;
		std::string tfileName = "temp.jpg";
		float *scores;
	private:	
        //图片预处理方式:只进行图像增强
		cv::Mat preProcess(cv::Mat img);
		//图片预处理方式2:进行图像裁剪和图像增强
		cv::Mat preProcess_cp(cv::Mat img);
		//图片预处理方式3:只进行图像裁剪
		cv::Mat preProcess_co(cv::Mat img);
        //从得分获取标签类别
		int getLabel(float *scores,int nClasses);
	public:
	    EyeDiseaseClassifier();
        
		EyeDiseaseClassifier(std::string classifierName_);
        //载入分类器模型
		bool loadModel(std::string model_dir,int nClasses_);

		//传入一张图片路径进行分类
		int Classify(std::string filedir,int preProcess_flag=PREPROCESS);

		//传入已经载入内存的图片进行分类
		int Classify(cv::Mat src,int preProcess_flag=PREPROCESS);//传入图像之前需要 先做预处理

		//释放分类器模型资源
		void releaseModel();
    
};