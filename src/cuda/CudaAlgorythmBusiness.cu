#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include "CudaAlgorythmBusiness.cuh"

////////////////////////////////////////////////////////////////////////////////
// Convolución de referencia en CPU filas y columnas
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Result, float *h_DataA, float *h_Kernel,
		int dataW, int dataH, int kernelR);

void convolutionColumnCPU(float *h_Result, float *h_DataA, float *h_Kernel,
		int dataW, int dataH, int kernelR);
////////////////////////////////////////////////////////////////////////////////
// Convolución de referencia para filas en CPU
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Result, float *h_Data, float *h_Kernel,
		int dataW, int dataH, int kernelR) {
	int x, y, k, d;
	float sum;

	for (y = 0; y < dataH; y++)
		for (x = 0; x < dataW; x++) {
			sum = 0;
			for (k = -kernelR; k <= kernelR; k++) {
				d = x + k;
				if (d < 0)
					d = 0;
				if (d >= dataW)
					d = dataW - 1;
				sum += h_Data[y * dataW + d] * h_Kernel[kernelR - k];
			}
			h_Result[y * dataW + x] = sum;
		}
}

////////////////////////////////////////////////////////////////////////////////
// Convolución de referencia para columnas en CPU
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Result, float *h_Data, float *h_Kernel,
		int dataW, int dataH, int kernelR) {
	int x, y, k, d;
	float sum;

	for (y = 0; y < dataH; y++)
		for (x = 0; x < dataW; x++) {
			sum = 0;
			for (k = -kernelR; k <= kernelR; k++) {
				d = y + k;
				if (d < 0)
					d = 0;
				if (d >= dataH)
					d = dataH - 1;
				sum += h_Data[d * dataW + x] * h_Kernel[kernelR - k];
			}
			h_Result[y * dataW + x] = sum;
		}
}

//Fast integer multiplication macro
#define IMUL(a, b) __mul24(a, b)

//Input data texture reference
texture<float, 2, cudaReadModeElementType> texData;

////////////////////////////////////////////////////////////////////////////////
// configuración del kernel
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 1
#define KERNEL_W      (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel1[KERNEL_W];
__device__ __constant__ float d_Kernel2[KERNEL_W];

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(float *d_Result, int dataW, int dataH) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum = 0;

		for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			sum += tex2D(texData, x + k, y) * d_Kernel1[KERNEL_RADIUS - k];

		d_Result[IMUL(iy, dataW) + ix] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(float *d_Result, int dataW, int dataH) {
	const int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const float x = (float) ix + 0.5f;
	const float y = (float) iy + 0.5f;

	if (ix < dataW && iy < dataH) {
		float sum = 0;

		for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			sum += tex2D(texData, x, y + k) * d_Kernel2[KERNEL_RADIUS - k];

		d_Result[IMUL(iy, dataW) + ix] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////

//Reduce problem size to have reasonable emulation time

int DATA_W;
int DATA_H;
int DATA_SIZE;
unsigned int width, height;
int KERNEL_SIZE = KERNEL_W * sizeof(float);

CudaAlgorythmBusiness::CudaAlgorythmBusiness() {
	// TODO Auto-generated constructor stub

}

CudaAlgorythmBusiness::~CudaAlgorythmBusiness() {
	// TODO Auto-generated destructor stub
}

float* CudaAlgorythmBusiness::GaussCUDA(float *imagen, int ancho, int alto,
		float *h_kernel1, float *h_kernel2) {

	float *h_DataA, *h_DataB, *h_ResultGPU, *h_Kernelx1,*h_Kernelx2;


	cudaArray *a_Data;

	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float> ();

	float *d_Result;

	double L1norm, rCPU, rGPU, sum_delta, sum_ref;

	int i, x, y;

	DATA_W = ancho;
	DATA_H = alto;
	DATA_SIZE = DATA_W * DATA_H * sizeof(float);

	h_DataA = (float *) malloc(DATA_SIZE);
	h_DataB = (float *) malloc(DATA_SIZE);
	h_ResultGPU = (float *) malloc(DATA_SIZE);
	h_Kernelx1 = (float *) malloc(KERNEL_SIZE);
	h_Kernelx2 = (float *) malloc(KERNEL_SIZE);
	CUDA_SAFE_CALL(cudaMallocArray(&a_Data, &floatTex, DATA_W, DATA_H));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Result, DATA_SIZE));

	h_Kernelx1 = h_kernel1;
	h_Kernelx2 = h_kernel2;

	h_DataA = imagen;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel1, h_Kernelx1, KERNEL_SIZE));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel2, h_Kernelx2, KERNEL_SIZE));
	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, h_DataA, DATA_SIZE,
					cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaBindTextureToArray(texData, a_Data));

	//el ancho del bloque debe ser multiplo del máximo de memorria coalescente en escritura
	//para escrituras coalescentes en convolutionRowGPU y convolutionColumnGP
	dim3 threadBlock(16, 12);
	dim3
			blockGrid(iDivUp(DATA_W, threadBlock.x),
					iDivUp(DATA_H, threadBlock.y));


	CUDA_SAFE_CALL(cudaThreadSynchronize());

	convolutionRowGPU<<<blockGrid, threadBlock>>>(
			d_Result,
			DATA_W,
			DATA_H
	);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpyToArray(a_Data, 0, 0, d_Result, DATA_SIZE,
					cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	convolutionColumnGPU<<<blockGrid, threadBlock>>>(
			d_Result,
			DATA_W,
			DATA_H
	);

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(
			cudaMemcpy(h_ResultGPU, d_Result, DATA_SIZE, cudaMemcpyDeviceToHost));

	convolutionRowCPU(h_DataB, h_DataA, h_Kernelx1, DATA_W, DATA_H,
			KERNEL_RADIUS);

	CUDA_SAFE_CALL(cudaUnbindTexture(texData));
	CUDA_SAFE_CALL(cudaFree(d_Result));
	CUDA_SAFE_CALL(cudaFreeArray(a_Data));
	free(h_DataB);
	free(h_Kernelx1);
	free(h_Kernelx2);

	return h_ResultGPU;
}
