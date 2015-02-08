/*
 * Cuda5StepConvolutionBusiness.h
 *
 *  Created on: 15/06/2012
 *      Author: jose
 */

#ifndef CUDA5STEPCONVOLUTIONBUSINESS_H_
#define CUDA5STEPCONVOLUTIONBUSINESS_H_

class Cuda5StepConvolutionBusiness {
public:
	Cuda5StepConvolutionBusiness();
	virtual ~Cuda5StepConvolutionBusiness();
	float* convolve(float *imagen, int ancho, int alto, float *h_kernel1,float *h_kernel2, int tamFilter);
	int iDivUp(int a, int b);
	//Round a / b to nearest lower integer value
	int iDivDown(int a, int b);
	//Align a to nearest higher multiple of b
	int iAlignUp(int a, int b);
	//Align a to nearest lower multiple of b
	int iAlignDown(int a, int b);
};

#endif /* CUDA5STEPCONVOLUTIONBUSINESS_H_ */
