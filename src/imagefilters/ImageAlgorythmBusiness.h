/*
 * ImageAlgorythmBusiness.h
 *
 *  Created on: 24/03/2012
 *      Author: jose
 */

#ifndef IMAGEALGORYTHMBUSINESS_H_
#define IMAGEALGORYTHMBUSINESS_H_

#include "QImage"
#include "./src/imagefilters/SobelImageFilter.h"
#include "./src/imagefilters/PrewittImageFilter.h"
#include "./src/imagefilters/SobelSquareImageFilter.h"
#include "./src/imagefilters/LaplaceImageFilter.h"
#include "./src/imagefilters/RobertCrossImageFilter.h"
#include "./src/imagefilters/CannyImageFilter.h"
#include "./src/imagefilters/LaplacianOfGaussianImageFilter.h"


class ImageAlgorythmBusiness {
public:
	ImageAlgorythmBusiness();
	virtual ~ImageAlgorythmBusiness();
	//Ejecución en CPU
	QImage SobelFilterCPU(QImage imagenEntrada,SobelImageFilter sobelFilter);
	QImage PrewittFilterCPU(QImage imagenEntrada,PrewittImageFilter prewittFilter);
	QImage LaplaceFilterCPU(QImage imagenEntrada,LaplaceImageFilter laplaceFilter);
	QImage RobertCrossFilterCPU(QImage imagenEntrada,RobertCrossImageFilter robertCrossFilter);
	QImage LaplacianOfGaussianFilterCPU(QImage imagenEntrada, LaplacianOfGaussianImageFilter laplacianOfGaussian);
	QImage CannyFilterCPU(QImage imagenEntrada,CannyImageFilter cannyFilter);
	QImage SobelSquareFilterCPU(QImage imagenEntrada,SobelSquareImageFilter sobelSquareFilter);

	//Ejecución GPU
	QImage SobelFilterGPU(QImage imagenEntrada);
	QImage PrewittFilterGPU(QImage imagenEntrada);
	QImage LaplaceFilterGPU(QImage imagenEntrada);
	QImage RobertCrossFilterGPU(QImage imagenEntrada);
	QImage LaplacianOfGaussianFilterGPU(QImage imagenEntrada,LaplacianOfGaussianImageFilter laplacianOfGaussianFilter);
	QImage SobelSquareFilterGPU(QImage imagenEntrada);
	//Fuciones auxiliares para negocio
	QImage grayScale(QImage imagenEntrada,int pad);
	QImage Gauss(QImage imagenEntrada, int radioGauss, double sigmaGauss);
	QImage convolutionCPU(int surceImageWidht,int sourceImageHeight,QImage paddedImage,ImageFilter imageFilter);
	float* fromQImageToMatrix(QImage imagenEntrada);
	float* fromQImageToMatrix(QImage imagenEntrada,float *puntero);
	QImage fromMatrixToQImage(float* matrix,int width,int height);
	float* generateGaussianKernels(float sigma, int gaussRadius);
	void manageExecutionTimeWriting(int executionTime,QString processName,QString mode,bool cpu);


	//Algoritmos en memoria compartida (shared memory)
	QImage SobelFilterGPUShared(QImage imagenEntrada);
	QImage SobelSquareFilterGPUShared(QImage imagenEntrada);
	QImage PrewittFilterGPUShared(QImage imagenEntrada);
	QImage RobertCrossFilterGPUShared(QImage imagenEntrada);
	QImage LaplaceFilterGPUShared(QImage imagenEntrada);
	QImage LaplacianOfGaussianFilterGPUShared(QImage imagenEntrada,LaplacianOfGaussianImageFilter laplacianOfGaussianFilter);

	//Métodos de prueba de los métodos de negocio de ITK.
	//QImage SobelItk(QImage imagenEntrada);
	QImage cannyGpuShared(QImage imagenEntrada,CannyImageFilter laplacianOfGaussianFilter);
};

#endif /* IMAGEALGORYTHMBUSINESS_H_ */
