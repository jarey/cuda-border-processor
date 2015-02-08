/*
 * CudaDiscreteGaussian.cu
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase encargada de la implementación en GPU memorial global de la creación del filtro gaussiano con los parámetros daos y del difuminado
 * de la imagen a partir de dicho filtro.
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaDiscreteGaussian.cuh"

/// Variables de textura para el trabajo con la imagen.
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> gaussTexRef;

CudaDiscreteGaussian::CudaDiscreteGaussian() {
	// TODO Auto-generated constructor stub

}

CudaDiscreteGaussian::~CudaDiscreteGaussian() {
	// TODO Auto-generated destructor stub
}

__global__ void calculateGaussianKernel(float *gaussKernel, const float sigma, int halfKernelWidth){

	//índices de pixels en este thread que confeccionarán la curva normal.
  int i = threadIdx.x - halfKernelWidth;
  extern __shared__ float s_gaussKernel[];
  __shared__ float sum;

  /// Este kernel debe reservar el número de kernel igual a fKernelWidth
  s_gaussKernel[threadIdx.x] = (__fdividef(1,(sqrtf(2*M_PI*sigma))))*expf((-1)*(__fdividef((i*i),(2*sigma*sigma))));
  __syncthreads();

  //El thread número 0, realiza la suma del array  gaussiano
  if (!threadIdx.x) {
    int th;
    sum = 0;
    for(th = 0; th<blockDim.x; th++) sum += s_gaussKernel[th];
  }

  __syncthreads();

  gaussKernel[threadIdx.x] = s_gaussKernel[threadIdx.x]/sum;

}

float* CudaDiscreteGaussian::cuda1DGaussianOperator(dim3 DimGrid, dim3 DimBlock, unsigned int width, float gaussianVariance){

  /// El ancho del kernel gaussiano debe ser impar.
  if (width < 1) width = 1;
  if (width%2 == 0) width--;
  short halfWidth = width >> 1;

  int kernelSize = width*sizeof(float);

  float *cudaGaussKernel;
  cudaMalloc((void**)&cudaGaussKernel,kernelSize);

  /// Se calcula el kernel gaussiano
  calculateGaussianKernel<<<DimGrid,DimBlock,kernelSize>>>(cudaGaussKernel, gaussianVariance, halfWidth);

  return cudaGaussKernel;
}
