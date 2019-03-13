#include "SparseClassifier.h"
EyeDiseaseClassifier::EyeDiseaseClassifier()
{
}
EyeDiseaseClassifier::EyeDiseaseClassifier(std::string classifierName_)
{
    classifierName = classifierName_;
    tfileName = classifierName+tfileName;
}

bool EyeDiseaseClassifier::loadModel(std::string model_dir,int nClasses_)
{

    nClasses = nClasses_;
    cnn=new Imagenet(2,VLEAKYRELU,nFeatures,nClasses,cudaDevice);
    scores = new float[nClasses];
    cnn->loadWeights(model_dir,-1);
    return false;
}
//图片预处理方式1
cv::Mat EyeDiseaseClassifier::preProcess(cv::Mat img)
{
    std::cout<<"preProcess 1"<<std::endl;
    cv::Size size = img.size();
    //计算scaleRadius
    float scale = 300;//Scale 300 seems to be sufficient; 500 and 1000 are overkill
    cv::Mat middleRow = img.rowRange(size.height/2,size.height/2+1).clone();
    int x[size.width],xMean=0;
    int r=0;//保存半径的长度
    for(int i=0;i<size.width;i++)
    {
        x[i] = (int)middleRow.at<cv::Vec3b>(0,i)[0]
                +(int)middleRow.at<cv::Vec3b>(0,i)[1]
                +(int)middleRow.at<cv::Vec3b>(0,i)[2];
        if(i>0)
        {
            xMean = (((float)xMean)+((float)x[i])/i)/(1.0+1.0/i);
        }
        else if(i==0)
            xMean = x[i];

    }
    int tMean = ((float)xMean)/10.0;
    for(int i=0;i<size.width;i++)
    {
        if(x[i]>tMean)
            r++;
    }
    //从直径计算半径
    r/=2;
    float s=scale*1.0/((float)r);
    cv::Mat img_resized ;
    cv::resize(img,img_resized,cv::Size(0,0),s,s);
    cv::Mat img_black = cv::Mat::zeros( img_resized.size(), CV_8UC3 );
    cv::circle(img_black,img_resized.size()/2,int(scale*0.9),cv::Scalar(1,1,1),-1,8,0);
    cv::Mat gaussian;
    cv::GaussianBlur(img_resized,gaussian,cv::Size(0,0),scale/30);
    cv::Mat add_weighted_img;
    cv::addWeighted(img_resized,4,gaussian,-4,128,add_weighted_img);
    cv::Mat one_mat(img_resized.size(),CV_8UC3,cv::Scalar(1,1,1));
    cv::Mat result = add_weighted_img.mul(img_black)+128*(one_mat-img_black);
    return result;			
}
//图片预处理方式2
cv::Mat EyeDiseaseClassifier::preProcess_cp(cv::Mat img)
{
    cv::Size size = img.size();
    //计算scaleRadius
    float scale = 300;//Scale 300 seems to be sufficient; 500 and 1000 are overkill
    cv::Mat middleRow = img.rowRange(size.height/2,size.height/2+1).clone();
    int x[size.width],xMean=0;
    int r=0;//保存半径的长度
    for(int i=0;i<size.width;i++)
    {
        x[i] = (int)middleRow.at<cv::Vec3b>(0,i)[0]
                +(int)middleRow.at<cv::Vec3b>(0,i)[1]
                +(int)middleRow.at<cv::Vec3b>(0,i)[2];
        if(i>0)
        {
            xMean = (((float)xMean)+((float)x[i])/i)/(1.0+1.0/i);
        }
        else if(i==0)
            xMean = x[i];

    }
    int tMean = ((float)xMean)/10.0+20;
    for(int i=0;i<size.width;i++)
    {
        if(x[i]>tMean)
            r++;
    }
    int width;
    int height;
    int xx,yy;
    if(size.width>r)
    {
        xx = (size.width-r)/2;
        width = r;
    }
    else
    {
        xx = 0;
        width = size.width;
    }

    if(size.height>r)
    {
        yy = (size.height-r)/2;
        height = r;
    }
    else
    {
        yy=0;
        height = size.height;
    }
    cv::Rect roi(xx+1, yy+1, width-1, height-1);			
    cv::Mat imgt = img(roi);
    //从直径计算半径
    r/=2;
    float s=scale*1.0/((float)r);
    cv::Mat img_resized ;
    cv::resize(imgt,img_resized,cv::Size(0,0),s,s);
    cv::Mat img_black = cv::Mat::zeros( img_resized.size(), CV_8UC3 );
    cv::circle(img_black,img_resized.size()/2,int(scale*0.9),cv::Scalar(1,1,1),-1,8,0);
    cv::Mat gaussian;
    cv::GaussianBlur(img_resized,gaussian,cv::Size(0,0),scale/30);
    cv::Mat add_weighted_img;
    cv::addWeighted(img_resized,4,gaussian,-4,128,add_weighted_img);
    cv::Mat one_mat(img_resized.size(),CV_8UC3,cv::Scalar(1,1,1));
    cv::Mat result = add_weighted_img.mul(img_black)+128*(one_mat-img_black);
    return result;			
}
//图片预处理方式3
cv::Mat EyeDiseaseClassifier::preProcess_co(cv::Mat img)
{
    cv::Size size = img.size();
    //计算scaleRadius
    float scale = 300;//Scale 300 seems to be sufficient; 500 and 1000 are overkill
    cv::Mat middleRow = img.rowRange(size.height/2,size.height/2+1).clone();
    int x[size.width],xMean=0;
    int r=0;//保存半径的长度
    for(int i=0;i<size.width;i++)
    {
        x[i] = (int)middleRow.at<cv::Vec3b>(0,i)[0]
                +(int)middleRow.at<cv::Vec3b>(0,i)[1]
                +(int)middleRow.at<cv::Vec3b>(0,i)[2];
        if(i>0)
        {
            xMean = (((float)xMean)+((float)x[i])/i)/(1.0+1.0/i);
        }
        else if(i==0)
            xMean = x[i];

    }
    int tMean = ((float)xMean)/10.0+20;
    for(int i=0;i<size.width;i++)
    {
        if(x[i]>tMean)
            r++;
    }
    
    int width;
    int height;
    int xx,yy;
    if(size.width>r)
    {
        xx = (size.width-r)/2;
        width = r;
    }
    else
    {
        xx = 0;
        width = size.width;
    }

    if(size.height>r)
    {
        yy = (size.height-r)/2;
        height = r;
    }
    else
    {
        yy=0;
        height = size.height;
    }
    cv::Rect roi(xx+1, yy+1, width-1, height-1);			
    cv::Mat imgt = img(roi);

    return imgt;			
}
//传入一张图片路径进行分类
int EyeDiseaseClassifier::Classify(std::string filedir,int preProcess_flag)
{
    cv::Mat img = cv::imread(filedir);
    return Classify(img,preProcess_flag);

}

//传入已经载入内存的图片进行分类
int EyeDiseaseClassifier::Classify(cv::Mat src,int preProcess_flag)//传入图像之前需要 先做预处理
{
    if(!src.data)
    {
        std::cout<<"no data!"<<std::endl;
        return -1;
    }
        
    //preProcess_flag = 1;
    cv::Mat preSrc;
    if(processFlag==PREPROCESS)
        preSrc = preProcess(src);
    else if(processFlag==PREPROCESS_CP)
        preSrc = preProcess_cp(src);
    else if(processFlag==PREPROCESS_CO)
        preSrc = preProcess_co(src);
    else return -1;				
    if(!preSrc.data)
        return -1;
    SpatiallySparseDataset testImg = KDRTestSet(tfileName,nClasses,preSrc);
    cnn->processDatasetRepeatTest(testImg,batchSize,6,scores);

    return getLabel(scores,nClasses);
}
//从得分获取标签类别
int EyeDiseaseClassifier::getLabel(float *scores,int nClasses)
{
    float max=-1;
    int label = -1;
    for(int i=0;i<nClasses;i++)
    {
        if(max<scores[i])
        {
            max = scores[i];
            label = i;
        }
    }
    return label;
}
//释放分类器模型资源
void EyeDiseaseClassifier::releaseModel()
{
    delete cnn;
    delete scores;
}