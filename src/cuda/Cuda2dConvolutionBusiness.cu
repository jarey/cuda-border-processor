/*
 * Cuda2dConvolutionBusiness.cpp
 *
 *  Created on: 05/04/2012
 *      Author: jose
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include "Cuda2dConvolutionBusiness.cuh"

Cuda2dConvolutionBusiness::Cuda2dConvolutionBusiness() {
	// TODO Auto-generated constructor stub
}

Cuda2dConvolutionBusiness::~Cuda2dConvolutionBusiness() {
	// TODO Auto-generated destructor stub
}
int Cuda2dConvolutionBusiness::iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int Cuda2dConvolutionBusiness::iAlignUp(int a, int b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}
//Fast integer multiplication macro
#define IMUL(a, b) __mul24(a, b)

//Input data texture reference
texture<float, 2, cudaReadModeElementType> textureData;

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolution2D(float *d_Result, float *d_Kernel11,
		float* d_Kernel21, int dataW, int dataH, int sqrtFilterTam) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum1 = 0;
		float sum2 = 0;
		for (int k = 0; k < sqrtFilterTam; k++) {
			for (int l = 0; l < sqrtFilterTam; l++) {
				sum1 += tex2D(textureData, (x) + (-k), (y) + (-l))
						* d_Kernel11[l * sqrtFilterTam + k];
				sum1 += tex2D(textureData, (x) + (-k), (y) + (-l))
						* d_Kernel21[l * sqrtFilterTam + k];
			}
		}

		d_Result[IMUL(iy, dataW) + ix] = sqrt(sum1 * sum1 + sum2 * sum2);
	}
}

__global__ void convolution2DLaplace(float *d_Result, float* d_Kernel11,
		int dataW, int dataH, int sqrtFilterTam) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum1 = 0;
		float sum2 = 0;
		for (int k = 0; k < sqrtFilterTam; k++) {
			for (int l = 0; l < sqrtFilterTam; l++) {
				sum1 += tex2D(textureData, (x) + (-k), (y) + (-l))
						* d_Kernel11[l * sqrtFilterTam + k];
			}
		}

		if (sum1 > 255) {
			sum1 = 255.0;
		} else if (sum1 < 0) {
			sum1 = 0;
		}

		d_Result[IMUL(iy, dataW) + ix] = sum1;
	}
}

__global__ void convolutionRowGaussGPU(float *d_Result, float* d_Kernel1,
		int dataW, int dataH, int KERNEL_RADIUS) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum = 0;

		for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			sum += tex2D(textureData, x + k, y) * d_Kernel1[KERNEL_RADIUS - k];

		d_Result[IMUL(iy, dataW) + ix] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGaussGPU(float *d_Result, float* d_Kernel1,
		int dataW, int dataH, int KERNEL_RADIUS) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum = 0;

		for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			sum += tex2D(textureData, x, y + k) * d_Kernel1[KERNEL_RADIUS - k];

		d_Result[IMUL(iy, dataW) + ix] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////

//Reduce problem size to have reasonable emulation time


float* Cuda2dConvolutionBusiness::Convolve(float *imagen, int ancho, int alto,
		float *h_kernel1, float *h_kernel2, int tamFilter) {

	float *h_DataA, *h_ResultGPU, *h_Kernelx1, *h_Kernelx2;
	int DATA_SIZE1 = 0;
	int DATA_W1 = 0;
	int DATA_H1 = 0;
	unsigned int width1 = 0;
	unsigned int height1 = 0;
	////////////////////////////////////////////////////////////////////////////////
	// Kernel configuration
	////////////////////////////////////////////////////////////////////////////////
	printf("");
	int KERNEL_TAM1 = tamFilter;
	int KERNEL_SIZE1 = KERNEL_TAM1 * sizeof(float);
	float *d_Kernel11;
	float *d_Kernel21;

	cudaArray *a_Data;

	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float> ();

	float *d_Result;

	int i, x, y;

	int sqrtFilterTam = sqrt(tamFilter);

	DATA_W1 = ancho;
	DATA_H1 = alto;
	DATA_SIZE1 = DATA_W1 * DATA_H1 * sizeof(float);

	h_DataA = (float *) malloc(DATA_SIZE1);
	h_ResultGPU = (float *) malloc(DATA_SIZE1);
	h_Kernelx1 = (float *) malloc(KERNEL_SIZE1);
	h_Kernelx2 = (float *) malloc(KERNEL_SIZE1);
	CUDA_SAFE_CALL(cudaMallocArray(&a_Data, &floatTex, DATA_W1, DATA_H1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Result, DATA_SIZE1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel11, KERNEL_SIZE1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel21, KERNEL_SIZE1));

	memcpy(h_Kernelx1, h_kernel1, tamFilter * sizeof(float));
	memcpy(h_Kernelx2, h_kernel2, tamFilter * sizeof(float));

	h_DataA = imagen;

	CUDA_SAFE_CALL(
			cudaMemcpy(d_Kernel11, h_Kernelx1, KERNEL_SIZE1,
					cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(
			cudaMemcpy(d_Kernel21, h_Kernelx1, KERNEL_SIZE1,
					cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, h_DataA, DATA_SIZE1,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(textureData, a_Data));
	//el ancho del bloque debe ser multiplo del máximo de memorria coalescente en escritura
	//para escrituras coalescentes en convolutionRowGPU y convolutionColumnGPU
	dim3 threadBlock(16, 12);
	dim3 blockGrid(this->iDivUp(DATA_W1, threadBlock.x),
			this->iDivUp(DATA_H1, threadBlock.y));

	CUDA_SAFE_CALL( cudaThreadSynchronize());
	convolution2D<<<blockGrid, threadBlock>>>(
			d_Result,
			d_Kernel11,
			d_Kernel21,
			DATA_W1,
			DATA_H1,
			sqrtFilterTam
	);

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpy(h_ResultGPU, d_Result, DATA_SIZE1,
					cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaUnbindTexture(textureData));
	CUDA_SAFE_CALL(cudaFree(d_Result));
	CUDA_SAFE_CALL(cudaFree(d_Kernel11));
	CUDA_SAFE_CALL(cudaFree(d_Kernel21));
	CUDA_SAFE_CALL(cudaFreeArray(a_Data));
	free(h_Kernelx1);
	free(h_Kernelx2);

	return h_ResultGPU;

}

float* Cuda2dConvolutionBusiness::ConvolveLaplace(float *imagen, int ancho,
		int alto, float *h_kernel1, int tamFilter) {

	float *h_DataA, *h_ResultGPU, *h_Kernelx1;
	int DATA_SIZE1 = 0;
	int DATA_W1 = 0;
	int DATA_H1 = 0;
	unsigned int width1 = 0;
	unsigned int height1 = 0;
	////////////////////////////////////////////////////////////////////////////////
	// Configuración
	////////////////////////////////////////////////////////////////////////////////
	printf("");
	int KERNEL_TAM1 = tamFilter;
	int KERNEL_SIZE1 = KERNEL_TAM1 * sizeof(float);
	float *d_Kernel11;

	cudaArray *a_Data;

	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float> ();

	float *d_Result;

	int i, x, y;

	int sqrtFilterTam = sqrt(tamFilter);

	DATA_W1 = ancho;
	DATA_H1 = alto;
	DATA_SIZE1 = DATA_W1 * DATA_H1 * sizeof(float);

	h_DataA = (float *) malloc(DATA_SIZE1);
	h_ResultGPU = (float *) malloc(DATA_SIZE1);
	h_Kernelx1 = (float *) malloc(KERNEL_SIZE1);
	CUDA_SAFE_CALL(cudaMallocArray(&a_Data, &floatTex, DATA_W1, DATA_H1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Result, DATA_SIZE1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel11, KERNEL_SIZE1));

	memcpy(h_Kernelx1, h_kernel1, tamFilter * sizeof(float));

	h_DataA = imagen;

	CUDA_SAFE_CALL(
			cudaMemcpy(d_Kernel11, h_Kernelx1, KERNEL_SIZE1,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, h_DataA, DATA_SIZE1,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(textureData, a_Data));

	//el ancho del bloque debe ser multiplo del máximo de memorria coalescente en escritura
	//para escrituras coalescentes en convolutionRowGPU y convolutionColumnGP
	dim3 threadBlock(16, 12);
	dim3 blockGrid(this->iDivUp(DATA_W1, threadBlock.x),
			this->iDivUp(DATA_H1, threadBlock.y));

	CUDA_SAFE_CALL( cudaThreadSynchronize());
	convolution2DLaplace<<<blockGrid, threadBlock>>>(
			d_Result,
			d_Kernel11,
			DATA_W1,
			DATA_H1,
			sqrtFilterTam
	);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpy(h_ResultGPU, d_Result, DATA_SIZE1,
					cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaUnbindTexture(textureData));
	CUDA_SAFE_CALL(cudaFree(d_Result));
	CUDA_SAFE_CALL(cudaFree(d_Kernel11));
	CUDA_SAFE_CALL(cudaFreeArray(a_Data));

	free(h_Kernelx1);

	return h_ResultGPU;
}

float* Cuda2dConvolutionBusiness::ConvolveGauss(float *imagen, int ancho,
		int alto, float *h_kernel1, int radiusFilter) {

	float *h_DataA, *h_ResultGPU, *h_Kernelx1;
	int DATA_SIZE1 = 0;
	int DATA_W1 = 0;
	int DATA_H1 = 0;
	unsigned int width1 = 0;
	unsigned int height1 = 0;
	////////////////////////////////////////////////////////////////////////////////
	// Configuración del kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("");
	int KERNEL_TAM1 = radiusFilter;
	int KERNEL_SIZE1 = (2 * KERNEL_TAM1 + 1) * sizeof(float);
	float *d_Kernel11;

	cudaArray *a_Data;
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float> ();
	float *d_Result;
	int i, x, y;
	DATA_W1 = ancho;
	DATA_H1 = alto;
	DATA_SIZE1 = DATA_W1 * DATA_H1 * sizeof(float);

	h_DataA = (float *) malloc(DATA_SIZE1);
	h_ResultGPU = (float *) malloc(DATA_SIZE1);
	h_Kernelx1 = (float *) malloc(KERNEL_SIZE1);
	CUDA_SAFE_CALL(cudaMallocArray(&a_Data, &floatTex, DATA_W1, DATA_H1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Result, DATA_SIZE1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel11, KERNEL_SIZE1));

	memcpy(h_Kernelx1, h_kernel1, KERNEL_SIZE1 * sizeof(float));

	h_DataA = imagen;

	CUDA_SAFE_CALL(
			cudaMemcpy(d_Kernel11, h_Kernelx1, KERNEL_SIZE1,
					cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, h_DataA, DATA_SIZE1,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(textureData, a_Data));
	//el ancho del bloque debe ser multiplo del máximo de memorria coalescente en escritura
	//para escrituras coalescentes en convolutionRowGPU y convolutionColumnGP
	dim3 threadBlock(16, 12);
	dim3 blockGrid(this->iDivUp(DATA_W1, threadBlock.x),
			this->iDivUp(DATA_H1, threadBlock.y));

	CUDA_SAFE_CALL( cudaThreadSynchronize());
	convolutionRowGaussGPU<<<blockGrid, threadBlock>>>(
			d_Result,
			d_Kernel11,
			DATA_W1,
			DATA_H1,
			KERNEL_TAM1
	);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, d_Result, DATA_SIZE1,
					cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	convolutionColumnGaussGPU<<<blockGrid, threadBlock>>>(
			d_Result,
			d_Kernel11,
			DATA_W1,
			DATA_H1,
			KERNEL_TAM1
	);

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpy(h_ResultGPU, d_Result, DATA_SIZE1,
					cudaMemcpyDeviceToHost));
;
	CUDA_SAFE_CALL(cudaUnbindTexture(textureData));
	CUDA_SAFE_CALL(cudaFree(d_Result));
	CUDA_SAFE_CALL(cudaFree(d_Kernel11));
	CUDA_SAFE_CALL(cudaFreeArray(a_Data));

	return h_ResultGPU;
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRow(float *d_Result, float *d_kernel,
		int kernelRadius, int dataW, int dataH) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum = 0;

		for (int k = -kernelRadius; k <= kernelRadius; k++) {
			sum += tex2D(textureData, x + k, y) * d_kernel[kernelRadius - k];
		}

		d_Result[IMUL(iy, dataW) + ix] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumn(float *d_Result, float *d_kernel,
		int kernelRadius, int dataW, int dataH) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum = 0;

		for (int k = -kernelRadius; k <= kernelRadius; k++) {
			sum += tex2D(textureData, x, y + k) * d_kernel[kernelRadius - k];
		}

		d_Result[IMUL(iy, dataW) + ix] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////

//Reduce problem size to have reasonable emulation time


float* Cuda2dConvolutionBusiness::Convolution(float *imagen, int ancho,
		int alto, float *h_kernel1, float *h_kernel2, int kernelTam) {

	float *h_DataA, *h_ResultGPU, *h_Kernelx1, *h_Kernelx2;
	int DATA_SIZE1;
	int DATA_W1;
	int DATA_H1;
	////////////////////////////////////////////////////////////////////////////////
	// Kernel configuration
	////////////////////////////////////////////////////////////////////////////////
	printf("");
	int KERNEL_TAM1 = kernelTam;
	int KERNEL_SIZE1 = KERNEL_TAM1 * sizeof(float);
	float *d_Kernel11;
	float *d_Kernel21;

	cudaArray *a_Data;

	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float> ();

	float *d_Result;

	int kernelRadius = (kernelTam - 1) / 2;

	DATA_W1 = ancho;
	DATA_H1 = alto;
	DATA_SIZE1 = DATA_W1 * DATA_H1 * sizeof(float);

	h_DataA = (float *) malloc(DATA_SIZE1);
	h_ResultGPU = (float *) malloc(DATA_SIZE1);
	h_Kernelx1 = (float *) malloc(KERNEL_SIZE1);
	h_Kernelx2 = (float *) malloc(KERNEL_SIZE1);
	CUDA_SAFE_CALL(cudaMallocArray(&a_Data, &floatTex, DATA_W1, DATA_H1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Result, DATA_SIZE1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel11, KERNEL_SIZE1));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel21, KERNEL_SIZE1));

	memcpy(h_Kernelx1, h_kernel1, kernelTam * sizeof(float));
	memcpy(h_Kernelx2, h_kernel2, kernelTam * sizeof(float));

	h_DataA = imagen;

	CUDA_SAFE_CALL(
			cudaMemcpy(d_Kernel11, h_Kernelx1, KERNEL_SIZE1,
					cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(
			cudaMemcpy(d_Kernel21, h_Kernelx1, KERNEL_SIZE1,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, h_DataA, DATA_SIZE1,
					cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaBindTextureToArray(textureData, a_Data));
	//el ancho del bloque debe ser multiplo del máximo de memorria coalescente en escritura
	//para escrituras coalescentes en convolutionRowGPU y convolutionColumnGP
	dim3 threadBlock(16, 12);
	dim3 blockGrid(this->iDivUp(DATA_W1, threadBlock.x),
			this->iDivUp(DATA_H1, threadBlock.y));

	CUDA_SAFE_CALL( cudaThreadSynchronize());

	convolutionRow<<<blockGrid, threadBlock>>>(d_Result,d_Kernel11,kernelRadius, DATA_W1, DATA_H1);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, d_Result, DATA_SIZE1,
					cudaMemcpyDeviceToDevice));
	convolutionColumn<<<blockGrid, threadBlock>>>(d_Result,d_Kernel21,kernelRadius, DATA_W1, DATA_H1);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpy(h_ResultGPU, d_Result, DATA_SIZE1,
					cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaUnbindTexture(textureData));
	CUDA_SAFE_CALL(cudaFree(d_Result));
	CUDA_SAFE_CALL(cudaFree(d_Kernel11));
	CUDA_SAFE_CALL(cudaFree(d_Kernel21));
	CUDA_SAFE_CALL(cudaFreeArray(a_Data));
	free(h_Kernelx1);
	free(h_Kernelx2);

	return h_ResultGPU;
}

