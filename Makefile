CC=g++
CFLAGS=--std=c++11 -O3 -fPIC
NVCC=nvcc
NVCCFLAGS=--std=c++11 -arch sm_60 -O3 -Xcompiler -fPIC

SLIBGPU = libsparseclassifiergpu.a
LIBGPU = libsparseclassifiergpu.so

OBJ=BatchProducer.o ConvolutionalLayer.o ConvolutionalTriangularLayer.o IndexLearnerLayer.o MaxPoolingLayer.o \
    MaxPoolingTriangularLayer.o NetworkArchitectures.o NetworkInNetworkLayer.o Picture.o Regions.o Rng.o SigmoidLayer.o \
	SoftmaxClassifier.o SparseConvNet.o SparseConvNetCUDA.o SpatiallySparseBatch.o SpatiallySparseBatchInterface.o \
	SpatiallySparseDataset.o SpatiallySparseLayer.o TerminalPoolingLayer.o cudaUtilities.o readImageToMat.o types.o \
	utilities.o vectorCUDA.o vectorHash.o OpenCVPicture.o SpatiallySparseDatasetUtil.o \
	Imagenet.o SparseClassifier.o ReallyConvolutionalLayer.o

LIBS += -L/media/cql/DATA1/Development/opencv/install33/lib \
        -lrt -lcublas -larmadillo -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

INCLUDEPATH += -I/media/cql/DATA1/Development/opencv/install33/include \
               -I/media/cql/DATA1/Development/opencv/install33/include/opencv2 \
               -I/media/cql/DATA1/Development/opencv/install33/include/opencv
#LIBS += -L/home/olsen305/opencv_install_option/lib \		

%.o: %.cpp 
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDEPATH)
%.o: %.cu 
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(INCLUDEPATH)

CBD: $(OBJ)  CBD.o
	$(NVCC) -o CBD $(OBJ) CBD.o $(LIBS) $(NVCCFLAGS)
#	rm *.o

$(SLIBGPU) : $(OBJ)  ReallyConvolutionalLayer.o
	rm -f $@
	ar cr $@ $(OBJ)
	rm -f $(OBJ)

$(LIBGPU) : $(OBJ)  ReallyConvolutionalLayer.o
	rm -f $@
	g++ -shared -o $@ $(OBJ)
	rm -f $(OBJ)

clean:
	rm *.o *.a *.so
#tags :
#	ctags -R *
