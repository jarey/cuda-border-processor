/*
 * Cuda5bStepConvolutionBusiness.cpp
 *
 *  Created on: 15/06/2012
 *      Author: jose
 */

#include "Cuda5bStepConvolutionBusiness.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil.h>

Cuda5bStepConvolutionBusiness::Cuda5bStepConvolutionBusiness() {
	// TODO Auto-generated constructor stub

}

Cuda5bStepConvolutionBusiness::~Cuda5bStepConvolutionBusiness() {
	// TODO Auto-generated destructor stub
}

//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL5b(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS5b 1
#define      KERNEL_W5b (2 * KERNEL_RADIUS5b + 1)

// Assuming ROW_TILE_Wb, KERNEL_RADIUS5b_ALIGNEDb and dataW
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowbGPU()
#define            ROW_TILE_Wb 128
#define KERNEL_RADIUS5b_ALIGNEDb 16

// Assuming COLUMN_TILE_Wb and dataW are multiples
// of coalescing granularity size, all global memory operations
// are coalesced in convolutionColumnbGPU()
#define COLUMN_TILE_Wb 16
#define COLUMN_TILE_Hb 48

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRowb(float *data, float *d_Kernel1b) {
	return data[KERNEL_RADIUS5b - i] * d_Kernel1b[i] + convolutionRowb<i - 1>(data);
}

template<> __device__ float convolutionRowb<-1>(float *data, float *d_Kernel1b) {
	return 0;
}

template<int i> __device__ float convolutionColumnb(float *data, float *d_Kernel2b) {
	return data[(KERNEL_RADIUS5b - i) * COLUMN_TILE_Wb] * d_Kernel2b[i] + convolutionColumnb<i - 1>(data);
}

template<> __device__ float convolutionColumnb<-1>(float *data, float *d_Kernel2b) {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowbGPU5b(float *d_Result, float *d_Data, float *d_Kernel1b, int dataW,
		int dataH) {
	//Data cache
	__shared__ float data[KERNEL_RADIUS5b + ROW_TILE_Wb + KERNEL_RADIUS5b];

	//Current tile and apron limits, relative to row start
	const int tileStart = IMUL5b(blockIdx.x, ROW_TILE_Wb);
	const int tileEnd = tileStart + ROW_TILE_Wb - 1;
	const int apronStart = tileStart - KERNEL_RADIUS5b;
	const int apronEnd = tileEnd + KERNEL_RADIUS5b;

	//Clamp tile and apron limits by image borders
	const int tileEndClamped = min(tileEnd, dataW - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataW - 1);

	//Row start index in d_Data[]
	const int rowStart = IMUL5b(blockIdx.y, dataW);

	//Aligned apron start. Assuming dataW and ROW_TILE_Wb are multiples
	//of half-warp size, rowStart + apronStartAligned is also a
	//multiple of half-warp size, thus having proper alignment
	//for coalesced d_Data[] read.
	const int apronStartAligned = tileStart - KERNEL_RADIUS5b_ALIGNEDb;

	const int loadPos = apronStartAligned + threadIdx.x;
	//Set the entire data cache contents
	//Load global memory values, if indices are within the image borders,
	//or initialize with zeroes otherwise
	if (loadPos >= apronStart) {
		const int smemPos = loadPos - apronStart;

		data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos
				<= apronEndClamped)) ? d_Data[rowStart + loadPos] : 0;
	}

	//Ensure the completness of the loading stage
	//because results, emitted by each thread depend on the data,
	//loaded by another threads
	__syncthreads();
	const int writePos = tileStart + threadIdx.x;
	//Assuming dataW and ROW_TILE_Wb are multiples of half-warp size,
	//rowStart + tileStart is also a multiple of half-warp size,
	//thus having proper alignment for coalesced d_Result[] write.
	if (writePos <= tileEndClamped) {
		const int smemPos = writePos - apronStart;
		float sum = 0;

#ifdef UNROLL_INNER
		sum = convolutionRowb<2 * KERNEL_RADIUS5b>(data + smemPos,d_Kernel1b);
#else
		for (int k = -KERNEL_RADIUS5b; k <= KERNEL_RADIUS5b; k++)
			sum += data[smemPos + k] * d_Kernel1b[KERNEL_RADIUS5b - k];
#endif

		d_Result[rowStart + writePos] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnbGPU5b(float *d_Result, float *d_Data, float *d_Kernel2b,
		int dataW, int dataH, int smemStride, int gmemStride) {
	//Data cache
	__shared__ float data[COLUMN_TILE_Wb * (KERNEL_RADIUS5b + COLUMN_TILE_Hb + KERNEL_RADIUS5b)];

	//Current tile and apron limits, in rows
	const int tileStart = IMUL5b(blockIdx.y, COLUMN_TILE_Hb);
	const int tileEnd = tileStart + COLUMN_TILE_Hb - 1;
	const int apronStart = tileStart - KERNEL_RADIUS5b;
	const int apronEnd = tileEnd + KERNEL_RADIUS5b;

	//Clamp tile and apron limits by image borders
	const int tileEndClamped = min(tileEnd, dataH - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int apronEndClamped = min(apronEnd, dataH - 1);

	//Current column index
	const int columnStart = IMUL5b(blockIdx.x, COLUMN_TILE_Wb) + threadIdx.x;

	//Shared and global memory indices for current column
	int smemPos = IMUL5b(threadIdx.y, COLUMN_TILE_Wb) + threadIdx.x;
	int gmemPos = IMUL5b(apronStart + threadIdx.y, dataW) + columnStart;
	//Cycle through the entire data cache
	//Load global memory values, if indices are within the image borders,
	//or initialize with zero otherwise
	for (int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y) {
		data[smemPos]
				= ((y >= apronStartClamped) && (y <= apronEndClamped)) ? d_Data[gmemPos]
						: 0;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

	//Ensure the completness of the loading stage
	//because results, emitted by each thread depend on the data,
	//loaded by another threads
	__syncthreads();
	//Shared and global memory indices for current column
	smemPos = IMUL5b(threadIdx.y + KERNEL_RADIUS5b, COLUMN_TILE_Wb) + threadIdx.x;
	gmemPos = IMUL5b(tileStart + threadIdx.y , dataW) + columnStart;
	//Cycle through the tile body, clamped by image borders
	//Calculate and output the results
	for (int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y) {
		float sum = 0;

#ifdef UNROLL_INNER
		sum = convolutionColumnb<2 * KERNEL_RADIUS5b>(data + smemPos, float *d_Kernel2b);
#else
		for (int k = -KERNEL_RADIUS5b; k <= KERNEL_RADIUS5b; k++)
			sum += data[smemPos + IMUL5b(k, COLUMN_TILE_Wb)]
					* d_Kernel2b[KERNEL_RADIUS5b - k];
#endif

		d_Result[gmemPos] = sum;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}
}

//******************************************************
//******************************************************


//PGMWORKING



////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int Cuda5bStepConvolutionBusiness::iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int Cuda5bStepConvolutionBusiness::iDivDown(int a, int b) {
	return a / b;
}

//Align a to nearest higher multiple of b
int Cuda5bStepConvolutionBusiness::iAlignUp(int a, int b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int Cuda5bStepConvolutionBusiness::iAlignDown(int a, int b) {
	return a - a % b;
}

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
//Global macro, controlling innermost convolution loop unrolling
#define UNROLL_INNER

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Image width5 should be aligned to maximum coalesced read/write size
//for best global memory performance in both row and column filter.




//Carry out dummy calculations before main computation loop
//in order to "warm up" the hardware/driver
#define WARMUP
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
float* Cuda5bStepConvolutionBusiness::convolve(float *imagen, int ancho,
		int alto, float *h_kernel1, float *h_kernel2, int tamFilter) {

	float *h_Kernel1,*h_Kernel2 , *h_DataA, *h_DataB, *h_ResultGPU;
	unsigned int width5, height5; // Image width5 and height5
	int DATA_W5;
	int DATA_H5;
	int DATA_SIZE5;
	int KERNEL_SIZE5 = tamFilter * sizeof(float);
	float *d_DataA, *d_DataB, *d_Temp;

	//Variables para los kernnels en device
	float *d_Kernel1b;
	float *d_Kernel2b;

	cudaArray *a_Data;

	float *d_Result;

	double rCPU,rGPUsum_delta, sum_ref, L1norm, gpuTime;

	int i,x,y;

	DATA_W5 = ancho;
	DATA_H5 = alto;
	DATA_SIZE5 = DATA_W5 * DATA_H5 * sizeof(float);

	//Realizar la carga de la imagen


	//printf("%i x %i\n", DATA_W5, DATA_H5);
	//printf("Initializing data...\n");
	h_Kernel1 = (float *) malloc(KERNEL_SIZE5);
	h_Kernel1 = (float *) malloc(KERNEL_SIZE5);
	h_DataA = (float *) malloc(DATA_SIZE5);
	h_DataB = (float *) malloc(DATA_SIZE5);
	h_ResultGPU = (float *) malloc(DATA_SIZE5);
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_DataA, DATA_SIZE5));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_DataB, DATA_SIZE5));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Temp, DATA_SIZE5));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel1b, KERNEL_SIZE5));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_Kernel2b, KERNEL_SIZE5));

	h_Kernel1 = h_kernel1;
	h_Kernel2 = h_kernel2;

	h_DataA = imagen;

//	float kernelSum = 0;
//	for (i = 0; i < KERNEL_W5b; i++) {
//		float dist = (float) (i - KERNEL_RADIUS5b) / (float) KERNEL_RADIUS5b;
//		h_Kernel1[i] = expf(-dist * dist / 2);
//		kernelSum += h_Kernel1[i];
//	}
//	for (i = 0; i < KERNEL_W5b; i++)
//		h_Kernel1[i] /= kernelSum;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel1b, h_Kernel1, KERNEL_SIZE5));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel2b, h_Kernel2, KERNEL_SIZE5));

	cudaMemcpy(d_DataA, h_DataA, DATA_SIZE5, cudaMemcpyHostToDevice);

	dim3 blockGridRows(iDivUp(DATA_W5, ROW_TILE_Wb), DATA_H5);
	dim3 blockGridColumns(iDivUp(DATA_W5, COLUMN_TILE_Wb),
			iDivUp(DATA_H5, COLUMN_TILE_Hb));
	dim3 threadBlockRows(KERNEL_RADIUS5b_ALIGNEDb + ROW_TILE_Wb + KERNEL_RADIUS5b);
	dim3 threadBlockColumns(COLUMN_TILE_Wb, 8);

	//printf("GPU convolution...\n");
	cudaThreadSynchronize();

	convolutionRowbGPU5b<<<blockGridRows, threadBlockRows>>>(
			d_DataB,
			d_DataA,
			d_Kernel1b,
			DATA_W5,
			DATA_H5
	);
	CUT_CHECK_ERROR("convolutionRowbGPU() execution failed\n");

	convolutionColumnbGPU5b<<<blockGridColumns, threadBlockColumns>>>(
			d_DataA,
			d_DataB,
			d_Kernel1b,
			DATA_W5,
			DATA_H5,
			COLUMN_TILE_Wb * threadBlockColumns.y,
			DATA_W5 * threadBlockColumns.y
	);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//printf("GPU convolution time : %f msec //%f Mpixels/sec\n", gpuTime,
	//		1e-6 * DATA_W5 * DATA_H5 / (gpuTime * 0.001));

	//printf("Reading back GPU results...\n");
	CUDA_SAFE_CALL(
			cudaMemcpy(h_ResultGPU, d_DataA, DATA_SIZE5, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_Temp));
	CUDA_SAFE_CALL(cudaFree(d_DataB));
	CUDA_SAFE_CALL(cudaFree(d_DataA));
	//free(h_ResultGPU);
	//free(h_DataB);
	//free(h_DataA);
	free(h_Kernel1);
	free(h_Kernel2);

	return h_ResultGPU;
}
