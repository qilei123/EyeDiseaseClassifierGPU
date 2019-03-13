#include "SpatiallySparseDatasetUtil.h"
#include<iostream>
#include<fstream>
#include<string>
#include "utilities.h"

SpatiallySparseDataset KDRTrainSet(std::string dirName) {
  SpatiallySparseDataset dataset;
  dataset.name="DR train minus val";
  std::cout << "Loading "<< dataset.name<<"\n";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=6;

  std::string imageName,line;
  int cl;
  std::ifstream file(dirName+"train.txt");
  while (std::getline(file, line)) {
	imageName = line.substr(0,line.size()-2);

	cl = std::stoi(line.substr(line.size()-1,1));

    std::string filename=dirName+"train/"+imageName;
	//std::cout<<filename<<std::endl;
    if(globVector(filename).size()==1) {
      OpenCVPicture*  pic = new OpenCVPicture(filename,-1,128,cl);
      //OpenCVPicture*  pic = new OpenCVPicture(filename,128,cl);
      dataset.pictures.push_back(pic);
    }
  }
  return dataset;
}
SpatiallySparseDataset KDRValidationSet(std::string dirName) {
  SpatiallySparseDataset dataset;
  dataset.name="DR val set";
  std::cout << "Loading "<< dataset.name<<"\n";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=6;

  std::string imageName,line;
  int cl;
  std::ifstream file(dirName+"val.txt");
  while (std::getline(file, line)) {
    imageName = line.substr(0,line.size()-2);

	cl = std::stoi(line.substr(line.size()-1,1));

    std::string filename=dirName+"val/"+imageName;
    
    if(globVector(filename).size()==1) {
      OpenCVPicture*  pic = new OpenCVPicture(filename,-1,128,cl);
      //OpenCVPicture*  pic = new OpenCVPicture(filename,128,cl);
      dataset.pictures.push_back(pic);
    }
  }
  return dataset;
}
SpatiallySparseDataset KDRTestSet(std::string dirNameTest) {
  SpatiallySparseDataset dataset;
  dataset.name="DR test set";
  std::cout << "Loading "<< dataset.name<<"\n";
  dataset.type=UNLABELEDBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=6;

  std::string imageName,line;
  std::ifstream file(dirNameTest+"test.txt");
  while (std::getline(file, line)) {
    std::string filename=dirNameTest+"test/"+line;
    if(globVector(filename).size()==1) {
      OpenCVPicture*  pic = new OpenCVPicture(filename,-1,128,0);
      //OpenCVPicture*  pic = new OpenCVPicture(filename,128,0);
      dataset.pictures.push_back(pic);
    }
  }
  return dataset;
}

SpatiallySparseDataset KDRTestSet(std::string filename,int nClasses_) {
  SpatiallySparseDataset dataset;
  dataset.name="DR test set";
  std::cout << "Loading "<< dataset.name<<"\n";
  dataset.type=UNLABELEDBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=nClasses_;
  OpenCVPicture*  pic = new OpenCVPicture(filename,-1,128,0);
  //OpenCVPicture*  pic = new OpenCVPicture(filename,128,0);
  dataset.pictures.push_back(pic);
  return dataset;
}

SpatiallySparseDataset KDRTestSet(std::string filename,int nClasses_,cv::Mat &preSrc) {
  SpatiallySparseDataset dataset;
  dataset.name="DR test set";
  std::cout << "Loading "<< dataset.name<<"\n";
  dataset.type=UNLABELEDBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=nClasses_;
  OpenCVPicture*  pic = new OpenCVPicture(preSrc,filename,-1,128,0);
  //OpenCVPicture*  pic = new OpenCVPicture(filename,128,0);
  dataset.pictures.push_back(pic);
  return dataset;
}

void distortImageColorDR(cv::Mat& mat, RNG& rng, float sigma0, float sigma1, float sigma2, float sigma3) { //Background color 128
  cv::Mat mat2=cv::Mat::zeros(mat.rows,mat.cols,CV_8UC(mat.channels()));
  std::vector<float> delta0(mat.channels());
  std::vector<float> delta1(mat.channels());
  std::vector<float> delta2(mat.channels());
  std::vector<float> delta3(mat.channels());
  for (int j=0;j<mat.channels();j++) {
    delta0[j]=rng.normal(0,sigma0);
    delta1[j]=rng.normal(0,sigma1);
    delta2[j]=rng.normal(0,sigma2);
    delta3[j]=rng.normal(0,sigma3);
  }
  int j=0;
  for (int y=0;y<mat.rows;++y) {
    for (int x=0;x<mat.cols;++x) {
      for (int i=0;i<mat.channels();++i) {
        int c=mat.ptr()[j]-128;
        mat2.ptr()[j]=128+std::max(-128,std::min(127,
                                                 (int)(c*(1.0+
                                                          delta0[i]+
                                                          delta1[i]*sin(c*3.1415926535/128)+ //delta1 \in [-1,1]
                                                          delta2[i]*(x-0.5*mat.cols)+
                                                          delta3[i]*(y-0.5*mat.rows)))));
        ++j;
      }
    }
  }
  mat=mat2;
}
