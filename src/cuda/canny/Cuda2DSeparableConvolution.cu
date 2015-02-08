/*
 * Cuda2DSeparableConvolution.cu
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase encargada de realizar la convolución separable en memoria compartida de una imagen.
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>

#include "Cuda2DSeparableConvolution.cuh"

Cuda2DSeparableConvolution::Cuda2DSeparableConvolution() {
	// TODO Auto-generated constructor stub
}

Cuda2DSeparableConvolution::~Cuda2DSeparableConvolution() {
	// TODO Auto-generated destructor stub
}


/// Texturas para almacenaje de imagen
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> convTexRef;


__global__ void kernel_1DConvolutionH_texture(float *output, int3 size, short halfkernelsize){
//Se usa memoria de textura para almacenar la imagen entrante

  float sum = 0;
  int2 pos;

  extern __shared__ float s_mask[];

  ///píxel a tratar dentro del hilo
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  ///píxel de salida.
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<((halfkernelsize<<1)+1)) s_mask[threadIdx.x] = tex1Dfetch(convTexRef,threadIdx.x);

  __syncthreads();

  for(int k=-halfkernelsize;k<(halfkernelsize+1);k++){
    sum += (tex1Dfetch(texRef, pixIdx + k * (((pos.x+k)>=0)*((pos.x+k)<size.x))) * s_mask[k+halfkernelsize]);
  }

  output[pixIdx] = sum;
}

__global__ void kernel_1DConvolutionV_texture(float *output, int3 size, short halfkernelsize){
	//Se usa memoria de textura para almacenar la imagen entrante

  float sum = 0;
  int2 pos;

  extern __shared__ float s_mask[];

  ///pixel de trabajo en este hilo
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  ///pixel de salida.
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<((halfkernelsize<<1)+1)) s_mask[threadIdx.x] = tex1Dfetch(convTexRef,threadIdx.x);

  __syncthreads();

  for(int k=-halfkernelsize;k<(halfkernelsize+1);k++){
    sum += (tex1Dfetch(texRef, pixIdx + (size.x*k) * (((pos.y+k)>=0)*((pos.y+k<size.y)))) * s_mask[k+halfkernelsize]);
  }

  output[pixIdx] = sum;
}

float* Cuda2DSeparableConvolution::cuda2DSeparableConvolution(dim3 DimGrid, dim3 DimBlock, const float *d_img, int width, int height, const float *d_kernelH, int sizeH, const float *d_kernelV, int sizeV){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  cudaFuncSetCacheConfig(kernel_1DConvolutionH_texture,cudaFuncCachePreferNone);
  cudaFuncSetCacheConfig(kernel_1DConvolutionV_texture,cudaFuncCachePreferNone);

  short halfKernelWidthH = sizeH >> 1;
  int kernelSizeH = sizeH*sizeof(float);
  short halfKernelWidthV = sizeV >> 1;
  int kernelSizeV = sizeV*sizeof(float);

  float *d_output;
  cudaMalloc((void**) &d_output, size.z*sizeof(float));

  /// se reserva memoria para la imagen
  float *d_tmpbuffer;
  cudaMalloc((void**) &d_tmpbuffer, size.z*sizeof(float));

  /// se asocia la textura al array.
  cudaBindTexture (NULL, convTexRef, d_kernelH);
  CUT_CHECK_ERROR("la asociación de la textura ha fallado.");

  /// se configura la txturas
  convTexRef.normalized = false;
  convTexRef.filterMode = cudaFilterModePoint;

  /// se asocia la textura al array
  cudaBindTexture (NULL, texRef, d_img);
  CUT_CHECK_ERROR("la asociación de la textura ha fallado.");

  /// se configura la textura,
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_1DConvolutionH_texture<<<DimGrid,DimBlock,kernelSizeH>>>(d_tmpbuffer,size,halfKernelWidthH);
  CUT_CHECK_ERROR("error en la ejecución de la primera convolución.");
  /// se asocia la textura al array
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(convTexRef);
  cudaBindTexture (NULL ,texRef, d_tmpbuffer);
  cudaBindTexture (NULL, convTexRef, d_kernelV);

  kernel_1DConvolutionV_texture<<<DimGrid,DimBlock,kernelSizeV>>>(d_output,size,halfKernelWidthV);
  CUT_CHECK_ERROR("error en la ejecución de la segunda convolución.");
  /// se libera la memoria.
  cudaFree(d_tmpbuffer);
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(convTexRef);
  CUT_CHECK_ERROR("la liberación de la memoria ha fallado");

  return(d_output);
}
