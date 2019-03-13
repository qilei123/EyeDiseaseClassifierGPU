#ifndef IMAGENET_H
#define IMAGENET_H

#include "SparseConvNet.h"
#define NET_TYPE 3
class Imagenet : public SparseConvNet
{
public:
    Imagenet (int dimension, ActivationFunction fn, int nInputFeatures, int nClasses, int cudaDevice=-1, int nTop=1);
};

#endif // IMAGENET_H
