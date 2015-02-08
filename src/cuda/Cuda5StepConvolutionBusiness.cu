/*
 * Cuda5StepConvolutionBusiness.cpp
 *
 *  Created on: 15/06/2012
 *      Author: jose
 */

#include "Cuda5StepConvolutionBusiness.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cutil.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <QTime>
#include "./src/common/Controlador.h"


Cuda5StepConvolutionBusiness::Cuda5StepConvolutionBusiness() {
	// TODO Auto-generated constructor stub

}

Cuda5StepConvolutionBusiness::~Cuda5StepConvolutionBusiness() {
	// TODO Auto-generated destructor stub
}

//Macro apra multiplicación rápida de enteros
#define IMUL5(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Configuración del kernel
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS5 1
#define      KERNEL_W5 (2 * KERNEL_RADIUS5 + 1)
__device__ __constant__ float d_Kernel1[KERNEL_W5];
__device__ __constant__ float d_Kernel2[KERNEL_W5];

// Assuming ROW_TILE_W, KERNEL_RADIUS5_ALIGNED and dataW
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W 128
#define KERNEL_RADIUS5_ALIGNED 16

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float *data) {
	return
	data[KERNEL_RADIUS5 - i] * d_Kernel1[i]
	+ convolutionRow<i - 1>(data);
}

template<> __device__ float convolutionRow<-1>(float *data) {
	return 0;
}

template<int i> __device__ float convolutionColumn(float *data) {
	return
	data[(KERNEL_RADIUS5 - i) * COLUMN_TILE_W] * d_Kernel2[i]
	+ convolutionColumn<i - 1>(data);
}

template<> __device__ float convolutionColumn<-1>(float *data) {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU5(float *d_Result, float *d_Data, int dataW,
		int dataH) {
	//Data cache
	__shared__ float data[KERNEL_RADIUS5 + ROW_TILE_W + KERNEL_RADIUS5];

	//Current tile and apron limits, relative to row start
	int tileStart = IMUL5(blockIdx.x, ROW_TILE_W);
	int tileEnd = tileStart + ROW_TILE_W - 1;
	int apronStart = tileStart - KERNEL_RADIUS5;
	int apronEnd = tileEnd + KERNEL_RADIUS5;

	//Clamp tile and apron limits by image borders
	int tileEndClamped = min(tileEnd, dataW - 1);
	int apronStartClamped = max(apronStart, 0);
	int apronEndClamped = min(apronEnd, dataW - 1);

	//Row start index in d_Data[]
	int rowStart = IMUL5(blockIdx.y, dataW);

	//Aligned apron start. Assuming dataW and ROW_TILE_W are multiples
	//of half-warp size, rowStart + apronStartAligned is also a
	//multiple of half-warp size, thus having proper alignment
	//for coalesced d_Data[] read.
	int apronStartAligned = tileStart - KERNEL_RADIUS5_ALIGNED;

	int loadPos = apronStartAligned + threadIdx.x;
	//Set the entire data cache contents
	//Load global memory values, if indices are within the image borders,
	//or initialize with zeroes otherwise
	if (loadPos >= apronStart) {
		int smemPos = loadPos - apronStart;

		data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos
				<= apronEndClamped)) ? d_Data[rowStart + loadPos] : 0;
	}

	//Ensure the completness of the loading stage
	//because results, emitted by each thread depend on the data,
	//loaded by another threads
	__syncthreads();
	int writePos = tileStart + threadIdx.x;
	//Assuming dataW and ROW_TILE_W are multiples of half-warp size,
	//rowStart + tileStart is also a multiple of half-warp size,
	//thus having proper alignment for coalesced d_Result[] write.
	if (writePos <= tileEndClamped) {
		const int smemPos = writePos - apronStart;
		float sum = 0;

#ifdef UNROLL_INNER
		sum = convolutionRow<2 * KERNEL_RADIUS5>(data + smemPos);
#else
		for (int k = -KERNEL_RADIUS5; k <= KERNEL_RADIUS5; k++)
			sum += data[smemPos + k] * d_Kernel1[KERNEL_RADIUS5 - k];
#endif

		d_Result[rowStart + writePos] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU5(float *d_Result, float *d_Data,
		int dataW, int dataH, int smemStride, int gmemStride) {
	//Data cache
	__shared__ float data[COLUMN_TILE_W * (KERNEL_RADIUS5 + COLUMN_TILE_H + KERNEL_RADIUS5)];

	//Current tile and apron limits, in rows
	int tileStart = IMUL5(blockIdx.y, COLUMN_TILE_H);
	int tileEnd = tileStart + COLUMN_TILE_H - 1;
	int apronStart = tileStart - KERNEL_RADIUS5;
	int apronEnd = tileEnd + KERNEL_RADIUS5;

	//Clamp tile and apron limits by image borders
	int tileEndClamped = min(tileEnd, dataH - 1);
	int apronStartClamped = max(apronStart, 0);
	int apronEndClamped = min(apronEnd, dataH - 1);

	//Current column index
	int columnStart = IMUL5(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

	//Shared and global memory indices for current column
	int smemPos = IMUL5(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
	int gmemPos = IMUL5(apronStart + threadIdx.y, dataW) + columnStart;
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
	smemPos = IMUL5(threadIdx.y + KERNEL_RADIUS5, COLUMN_TILE_W) + threadIdx.x;
	gmemPos = IMUL5(tileStart + threadIdx.y , dataW) + columnStart;
	//Cycle through the tile body, clamped by image borders
	//Calculate and output the results
	for (int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y) {
		float sum = 0;

#ifdef UNROLL_INNER
		sum = convolutionColumn<2 * KERNEL_RADIUS5>(data + smemPos);
#else
		for (int k = -KERNEL_RADIUS5; k <= KERNEL_RADIUS5; k++)
			sum += data[smemPos + IMUL5(k, COLUMN_TILE_W)]
					* d_Kernel2[KERNEL_RADIUS5 - k];
#endif

		d_Result[gmemPos] = sum;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}
}

//******************************************************
//******************************************************


//PGMWORKING


unsigned int width5, height5; // Image width5 and height5
////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int Cuda5StepConvolutionBusiness::iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int Cuda5StepConvolutionBusiness::iDivDown(int a, int b) {
	return a / b;
}

//Align a to nearest higher multiple of b
int Cuda5StepConvolutionBusiness::iAlignUp(int a, int b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int Cuda5StepConvolutionBusiness::iAlignDown(int a, int b) {
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


int DATA_W5;
int DATA_H5;
int DATA_SIZE5;
int KERNEL_SIZE5 = KERNEL_W5 * sizeof(float);

//Carry out dummy calculations before main computation loop
//in order to "warm up" the hardware/driver
#define WARMUP
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
float* Cuda5StepConvolutionBusiness::convolve(float *imagen, int ancho,
		int alto, float *h_kernel1, float *h_kernel2, int tamFilter) {

	Controlador *controlador = Controlador::Instance();
	float  *h_DataB, *h_ResultGPU;
	float *d_DataA, *d_DataB;
	double rCPU,rGPUsum_delta, sum_ref, L1norm, gpuTime;

	DATA_W5 = ancho;
	DATA_H5 = alto;
	DATA_SIZE5 = DATA_W5 * DATA_H5 * sizeof(float);
	//Realizar la carga de la imagen
	h_DataB = (float *) malloc(DATA_SIZE5);
	h_ResultGPU = (float *) malloc(DATA_SIZE5);
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_DataA, DATA_SIZE5));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_DataB, DATA_SIZE5));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel1, h_kernel1, KERNEL_SIZE5));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel2, h_kernel2, KERNEL_SIZE5));
	cudaMemcpy(d_DataA, imagen, DATA_SIZE5, cudaMemcpyHostToDevice);

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	dim3 blockGridRows(iDivUp(DATA_W5, ROW_TILE_W), DATA_H5);
	dim3 blockGridColumns(iDivUp(DATA_W5, COLUMN_TILE_W),iDivUp(DATA_H5, COLUMN_TILE_H));
	dim3 threadBlockRows(KERNEL_RADIUS5_ALIGNED + ROW_TILE_W + KERNEL_RADIUS5);
	dim3 threadBlockColumns(COLUMN_TILE_W, 8);

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	convolutionRowGPU5<<<blockGridRows, threadBlockRows>>>(
			d_DataB,
			d_DataA,
			DATA_W5,
			DATA_H5
	);

	CUDA_SAFE_CALL(cudaThreadSynchronize());


	convolutionColumnGPU5<<<blockGridColumns, threadBlockColumns>>>(
			d_DataA,
			d_DataB,
			DATA_W5,
			DATA_H5,
			COLUMN_TILE_W * threadBlockColumns.y,
			DATA_W5 * threadBlockColumns.y
	);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	int elapsed=time->elapsed();
	controlador->setGpuExecutionTime(elapsed+controlador->getGpuExecutionTime());

	CUDA_SAFE_CALL(cudaMemcpy(h_ResultGPU, d_DataA, DATA_SIZE5, cudaMemcpyDeviceToHost));

	//liberación de memoria en cuda
	CUDA_SAFE_CALL(cudaFree(d_DataB));
	CUDA_SAFE_CALL(cudaFree(d_DataA));

	//free(h_ResultGPU);
	free(h_DataB);

	return h_ResultGPU;
}

