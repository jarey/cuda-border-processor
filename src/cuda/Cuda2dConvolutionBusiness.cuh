/*
 * Cuda2dConvolutionBusiness.h
 *
 *  Created on: 05/04/2012
 *      Author: jose
 */

#ifndef CUDA2DCONVOLUTIONBUSINESS_H_
#define CUDA2DCONVOLUTIONBUSINESS_H_

class Cuda2dConvolutionBusiness {
public:
	Cuda2dConvolutionBusiness();
	virtual ~Cuda2dConvolutionBusiness();
	float* Convolve(float *imagen, int ancho, int alto,float *kernel1,float *kernel2,int tamFilter);
	float* ConvolveLaplace(float *imagen, int ancho, int alto,float *kernel1,int tamFilter);
	int iDivUp(int a, int b);	//Align a to nearest higher multiple of b
	int iAlignUp(int a, int b);
	float* ConvolveGauss(float *imagen, int ancho,int alto, float *h_kernel1, int radiusFilter);
	float* Convolution(float *imagen, int ancho, int alto,float *h_kernel1, float *h_kernel2,int kernelTam);
};

#endif /* CUDA2DCONVOLUTIONBUSINESS_H_ */
