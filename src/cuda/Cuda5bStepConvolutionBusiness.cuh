/*
 * Cuda5StepConvolutionBusiness.h
 *
 *  Created on: 15/06/2012
 *      Author: jose
 */

#ifndef CUDA5BSTEPCONVOLUTIONBUSINESS_H_
#define CUDA5BSTEPCONVOLUTIONBUSINESS_H_

class Cuda5bStepConvolutionBusiness {
public:
	Cuda5bStepConvolutionBusiness();
	virtual ~Cuda5bStepConvolutionBusiness();
	float* convolve(float *imagen, int ancho, int alto, float *h_kernel1,float *h_kernel2, int tamFilter);
	int iDivUp(int a, int b);
	//Round a / b to nearest lower integer value
	int iDivDown(int a, int b);
	//Align a to nearest higher multiple of b
	int iAlignUp(int a, int b);
	//Align a to nearest lower multiple of b
	int iAlignDown(int a, int b);
};

#endif /* CUDA5BSTEPCONVOLUTIONBUSINESS_H_ */
