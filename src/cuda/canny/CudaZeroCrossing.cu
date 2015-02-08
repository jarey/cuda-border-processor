/*
 * CudaZeroCrossing.cu
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: Clase que implementa en GPU el cruce a 0 de una imagen. Paso incluido en la implementación dada para el proceso de Canny
 * en GPU memoria compartida en lugar del proceso de supresión de no-maximos.
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaZeroCrossing.cuh"


/// Texura para almacenamiento de imagen.
texture<float, 1, cudaReadModeElementType> texRef;


CudaZeroCrossing::CudaZeroCrossing() {
	// TODO Auto-generated constructor stub

}

CudaZeroCrossing::~CudaZeroCrossing() {
	// TODO Auto-generated destructor stub
}


__global__ void kernel_zerocrossing(float* image, int3 size){

  //Kernel para ejecución del cruce a 0. Se implementa usando obtención de datos a partir de memoria de textura.

  float  pixel;
  float4 cross;
  float res = 0;

  int pixIdx = blockDim.x * blockIdx.x + threadIdx.x;
  pixel = tex1Dfetch(texRef,pixIdx);

  ///índice para el pixel.
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  cross.x = tex1Dfetch(texRef,pixIdx-(pos.x>0));
  cross.y = tex1Dfetch(texRef,pixIdx-(size.x*(pos.y>0)));
  cross.z = tex1Dfetch(texRef,pixIdx+(pos.x<(size.x-1)));
  cross.w = tex1Dfetch(texRef,pixIdx+(size.x*(pos.y<(size.y-1))));

  res = (((pixel*cross.x)<=0) *\
      (fabsf(pixel) < fabsf(cross.x)));

  res = res || ((((pixel*cross.y)<=0)) *\
      (fabsf(pixel) < fabsf(cross.y)));

  res = res || ((((pixel*cross.z)<=0)) *\
      (fabsf(pixel) <= fabsf(cross.z)));

  res = res || ((((pixel*cross.w)<=0)) *\
      (fabsf(pixel) <= fabsf(cross.w)));

  image[pixIdx] = res;

}

float* CudaZeroCrossing::cudaZeroCrossing(dim3 DimGrid, dim3 DimBlock, float *d_input, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  cudaFuncSetCacheConfig(kernel_zerocrossing,cudaFuncCachePreferNone);

  float *d_img;
  cudaMalloc((void**) &d_img, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("La creación de la imagen en device ha fallado.");

  /// Se asocia la textura a un array de cuda.
  cudaBindTexture (NULL, texRef, d_input);
  CUT_CHECK_ERROR("La asociación d ela textura ha fallado.");

  /// Se configuran los atributos de la textura modo punto.
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_zerocrossing<<<DimGrid,DimBlock>>>(d_img, size);

  ///Se libera memoria.
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("La desasociación de la textura ha fallado.");

  return(d_img);

}
