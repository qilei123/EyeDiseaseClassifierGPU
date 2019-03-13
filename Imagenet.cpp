#include "Imagenet.h"


Imagenet::Imagenet(int dimension,
                   ActivationFunction fn,
                   int nInputFeatures,
                   int nClasses,
                   int cudaDevice,
                   int nTop)
  : SparseConvNet(dimension,nInputFeatures, nClasses, nTop)
{
    int net_type = NET_TYPE;
    if(net_type==1)
    {
        addLeNetLayerPOFMP( 32,5,1,3,1.8,fn);
        addLeNetLayerPOFMP( 64,3,1,3,1.8,fn);
        addLeNetLayerPOFMP( 96,3,1,3,1.8,fn);
        addLeNetLayerPOFMP(128,3,1,3,1.8,fn);
        addLeNetLayerPOFMP(160,3,1,3,1.8,fn);
        addLeNetLayerPOFMP(192,3,1,3,1.8,fn);
        addLeNetLayerPOFMP(224,3,1,3,1.8,fn);
        addLeNetLayerPOFMP(256,3,1,3,1.8,fn,32.0/256);
        addLeNetLayerPOFMP(288,3,1,2,1.5,fn,32.0/288);
        addLeNetLayerMP   (320,2,1,1,1  ,fn,64.0/320);
        addLeNetLayerMP   (356,1,1,1,1  ,fn,64.0/356);
        
    }
    else if (net_type==2)
    {
        addLeNetLayerPOFMP( 32,5,1,3,1.5,fn);
        addLeNetLayerPOFMP( 64,3,1,3,1.5,fn);
        addLeNetLayerPOFMP( 96,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(128,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(160,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(192,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(224,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(256,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(288,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(320,3,1,3,1.5,fn);
        addLeNetLayerPOFMP(352,3,1,3,1.6,fn,32.0/352);
        addLeNetLayerPOFMP(384,3,1,2,1.5,fn,32.0/384);
        addLeNetLayerMP(416,2,1,1,1,fn,64.0/416);
        addLeNetLayerMP(448,1,1,1,1,fn,64.0/448);        
    }
    else if (net_type==3)
    {
        for (int i=1; i <= 7; i++)
        {
            addLeNetLayerMP(32 * i, 3, 1, 1, 1, fn, 0.0f);
            addLeNetLayerMP(32 * i, 3, 1, 3, 2, fn, 0.0f);
        }
        addLeNetLayerMP(32 * 9, 2, 1, 1, 1, fn);
        addLeNetLayerMP(32 * 9, 2, 1, 1, 1, fn);
        addLeNetLayerMP(32 * 10, 1, 1, 1, 1, fn);
    }

    addSoftmaxLayer();
}
