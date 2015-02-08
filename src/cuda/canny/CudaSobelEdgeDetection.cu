/*
 * CudaSobelEdgeDetection.cu
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase que implementa el filtrado a través de la ventana de sobel de forma separable y en memoria compartida en la GPU.
 */


#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaSobelEdgeDetection.cuh"

#define THREADS_PER_BLOCK 256

/// Textura para alojar la imagen.
texture<float, 1, cudaReadModeElementType> texRef;

CudaSobelEdgeDetection::CudaSobelEdgeDetection() {
	// TODO Auto-generated constructor stub

}

CudaSobelEdgeDetection::~CudaSobelEdgeDetection() {
	// TODO Auto-generated destructor stub
}


__global__ void kernel_2DSobel(float *Magnitude, float *Direction, int3 size){
	//Versión del procesamiento de sobel para ejecutar de forma simplificada el cálculo en X y en Y de los valores de gradiente y dirección del
	//mismo en la imagen.

  float2 g_i;
  g_i.x = g_i.y = 0;
  float4 diagonal;
  float4 cross;

  /// ïndice del pixel a tratar en este thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  /// Índice del pixel de salida
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  ///Se ignoran los bordes de la imagen.
  if ((pos.x) && ((size.x-1)-pos.x) && (pos.y) && ((size.y-1)-pos.y)){

 //Se realiza el guardado de los vecinos del pixel tratao, ya que serán empleado más de una vez.
    diagonal.x = tex1Dfetch(texRef,(pixIdx-size.x-1));
    diagonal.y = tex1Dfetch(texRef,(pixIdx-size.x+1));
    diagonal.z = tex1Dfetch(texRef,(pixIdx+size.x-1));
    diagonal.w = tex1Dfetch(texRef,(pixIdx+size.x+1));
    cross.x = tex1Dfetch(texRef,(pixIdx-size.x));
    cross.y = tex1Dfetch(texRef,(pixIdx+size.x));
    cross.z = tex1Dfetch(texRef,(pixIdx-1));
    cross.w = tex1Dfetch(texRef,(pixIdx+1));

    /// SobelX
    g_i.x -= (diagonal.x+cross.z+cross.z+diagonal.z);
    g_i.x += (diagonal.y+cross.w+cross.w+diagonal.w);

    /// SobelY
    g_i.y -= (diagonal.z+cross.y+cross.y+diagonal.w);
    g_i.y += (diagonal.x+cross.x+cross.x+diagonal.y);

  }

  Magnitude[pixIdx] = sqrtf((g_i.x*g_i.x) + (g_i.y*g_i.y));

  Direction[pixIdx] = (g_i.x != 0)*(atanf(__fdividef(g_i.y,g_i.x)));

}

Tgrad* CudaSobelEdgeDetection::cudaSobel(Tgrad *d_gradient, const float *d_img, int width, int height){
  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int blocksPerGrid = ((size.z) + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
  dim3 DimBlock(THREADS_PER_BLOCK,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  /// Reserva de memoria para las matrices de gradiente y direccińo del gradiente.
  cudaMalloc((void**) &d_gradient->Magnitude, size.z*sizeof(float));
  cudaMalloc((void**) &d_gradient->Direction, size.z*sizeof(float));

  float *image;
  CUDA_SAFE_CALL(cudaMalloc((void **) &image, size.z*sizeof(float)));
  cudaMemcpy(image, d_img, size.z*sizeof(float), cudaMemcpyHostToDevice);
  //Se asocia la imagen a la textura
  cudaBindTexture(NULL,texRef, image);
  CUT_CHECK_ERROR("La asociación de la textura ha fallado.");

  /// Se configura la textura, modo punto.
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;
  kernel_2DSobel<<<DimGrid,DimBlock>>>(d_gradient->Magnitude, d_gradient->Direction, size);

  //Se desasocia la imagen a la textura.
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("La liberación de la textura ha fallado.");

  return(d_gradient);
}



float* CudaSobelEdgeDetection::cudaSobelM(Tgrad *d_gradient,float *d_img, int width, int height){
  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int blocksPerGrid = ((size.z) + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
  dim3 DimBlock(THREADS_PER_BLOCK,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  /// Reserva de memoria para las matrices de gradiente y dirección de gradiente.
  cudaMalloc((void**) &d_gradient->Magnitude, size.z*sizeof(float));
  cudaMalloc((void**) &d_gradient->Direction, size.z*sizeof(float));

  float *image;
  CUDA_SAFE_CALL(cudaMalloc((void **) &image, size.z*sizeof(float)));
  cudaMemcpy(image, d_img, size.z*sizeof(float), cudaMemcpyHostToDevice);
  //Se asocia la imagen a la textura
  cudaBindTexture(NULL,texRef, image);
  CUT_CHECK_ERROR("La asociacion de la texura ha fallado.");

  /// Se configura la textura, modo punto.
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;
  kernel_2DSobel<<<DimGrid,DimBlock>>>(d_gradient->Magnitude, d_gradient->Direction, size);

  cudaUnbindTexture(texRef);

  float *retorno;
  //Reservo memoria para el retorno.
  retorno = (float*) malloc(size.z*sizeof(float));
  cudaMemcpy(retorno,d_gradient->Magnitude, size.z*sizeof(float), cudaMemcpyDeviceToHost);
  CUT_CHECK_ERROR("La liberación de la memoria ha fallado.");

  return retorno;
}
