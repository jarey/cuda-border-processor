/*
 * CudaConvolutionSeparableBusiness.h
 *
 *  Created on: 07/04/2012
 *      Author: jose
 */

#ifndef CUDACONVOLUTIONSEPARABLEBUSINESS_H_
#define CUDACONVOLUTIONSEPARABLEBUSINESS_H_

class CudaConvolutionSeparableBusiness {
public:
	CudaConvolutionSeparableBusiness();
	virtual ~CudaConvolutionSeparableBusiness();
	float* Convolve(float *imagen, int ancho, int alto,float *h_kernel1, float *h_kernel2, int tamFilter);
	float* ConvolveLaplace(float *imagen, int ancho,int alto, float *h_kernel1, int tamFilter);
	int iDivUp(int a, int b);
	//alineamiento a multiplo más cercano de B
	int iAlignUp(int a, int b);
};

#endif /* CUDACONVOLUTIONSEPARABLEBUSINESS_H_ */
