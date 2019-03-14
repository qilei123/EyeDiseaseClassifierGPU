#include "SparseClassifier.h"
#include <time.h>
int main(int argc, char *argv[]) {
	std::string store_path("./models/");

	std::string baseName;
	if(NET_TYPE==3)
		baseName=store_path+"kaggleDiabeticRetinopathy3_epoch-50.cnn";
	else if(NET_TYPE==2)
		baseName=store_path+"kaggleDiabeticRetinopathy2_epoch-50.cnn";
	else if(NET_TYPE==1)
		baseName=store_path+"kaggleDiabeticRetinopathy1_epoch-50.cnn";
	EyeDiseaseClassifier *edc = new EyeDiseaseClassifier();
	//edc->initialModel(baseName,97,3,0);
	edc->loadModel(baseName,5);
	clock_t start, finish;
	start = clock();
	int i=0;
	int result = -1;
	//while(true)
	//{
	//i++;
	std::cout<<argv[1]<<std::endl;
	result=edc->Classify("./921607893.jpg",2);
	std::cout<<result<<std::endl;

	finish = clock();
	double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout<<duration<<std::endl;
	edc->releaseModel();
	return result;
}
