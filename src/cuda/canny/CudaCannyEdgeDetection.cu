/*
 * CudaCannyEdgeDetection.cu
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase encargada de realizar el proceso de canny, el proceso de histéresis.
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaCannyEdgeDetection.cuh"


/// Texturas para el almacenamiento de información en el algoritmo
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<float, 1, cudaReadModeElementType> hysTexRef;
texture<float, 1, cudaReadModeElementType> der_texRef;


CudaCannyEdgeDetection::CudaCannyEdgeDetection() {
	// TODO Auto-generated constructor stub

}

CudaCannyEdgeDetection::~CudaCannyEdgeDetection() {
	// TODO Auto-generated destructor stub
}


__global__ void kernel_Compute2ndDerivativePos(float *Magnitude, int3 size){
	//Este kernel recibe la imagen difuminada y su segunda derivada y retorna la magnitud
	//del gradiente en cada punto de la imagen.

  /// ïndice del pixel a procesar en este thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  /// Índice del pixel de salida.
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  float4 cross_Lvv;
  float4 cross_L;

  float Lx = 0,
        Ly = 0,
        Lvvx = 0,
        Lvvy = 0,
        gradMag;

  cross_L.x = tex1Dfetch(texRef,(pixIdx-(size.x*(pos.y>0))));
  cross_L.y = tex1Dfetch(texRef,(pixIdx+(size.x*(pos.y<(size.y-1)))));
  cross_L.z = tex1Dfetch(texRef,(pixIdx-(pos.x>0)));
  cross_L.w = tex1Dfetch(texRef,(pixIdx+(pos.x<(size.x-1))));
  cross_Lvv.x = tex1Dfetch(der_texRef,(pixIdx-(size.x*(pos.y>0))));
  cross_Lvv.y = tex1Dfetch(der_texRef,(pixIdx+(size.x*(pos.y<(size.y-1)))));
  cross_Lvv.z = tex1Dfetch(der_texRef,(pixIdx-(pos.x>0)));
  cross_Lvv.w = tex1Dfetch(der_texRef,(pixIdx+(pos.x<(size.x-1))));

  Lx = (-0.5*cross_L.z) + (0.5*cross_L.w);
  Ly = (0.5*cross_L.x) - (0.5*cross_L.y);

  Lvvx = (-0.5*cross_Lvv.z) + (0.5*cross_Lvv.w);
  Lvvy = (0.5*cross_Lvv.x) - (0.5*cross_Lvv.y);

  gradMag = sqrt((Lx*Lx)+(Ly*Ly));

  Magnitude[pixIdx] = (((Lvvx*(Lx/gradMag)+Lvvy*(Ly/gradMag))<=0)*gradMag);

}

float* CudaCannyEdgeDetection::cuda2ndDerivativePos(dim3 DimGrid, dim3 DimBlock, const float *d_input, const float *d_Lvv, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  cudaFuncSetCacheConfig(kernel_Compute2ndDerivativePos,cudaFuncCachePreferNone);

 /// Reserva de memoria para la imagen.
  float * d_mag;
  cudaMalloc((void**) &d_mag, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("La creación de la imagen ha fallado.");

  /// Asociación de la textura al array
  cudaBindTexture (NULL, texRef, d_input);
  cudaBindTexture (NULL, der_texRef, d_Lvv);
  CUT_CHECK_ERROR("La asociación de la textura ha fallado.");

  /// Se configura la textura de lado del host.
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_Compute2ndDerivativePos<<<DimGrid,DimBlock>>>(d_mag, size);

  /// Se desasocia la textura y libera memoria.
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(der_texRef);
  CUT_CHECK_ERROR("La desasociación de la textura ha fallado.");

  return(d_mag);

}


__global__ void kernel_Compute2ndDerivative(float *Lvv, int3 size){
	//Este kernel toma como entrada la imagen difuminada y devuelve la segunda derivada de la misma
	//En cada punto de la imagen.

  /// Índice del pixel a tratar
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  /// Píxel de salida.
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  float4 diagonal;
  float4 cross;
  float pixel = tex1Dfetch(texRef,(pixIdx));

  float Lx = 0,
        Ly = 0,
        Lxx = 0,
        Lxy = 0,
        Lyy = 0;

  //Los vecinos del pixel siendo procesado se guardan ya que su utilización se realiza más de una vez.
  diagonal.x = tex1Dfetch(texRef,(pixIdx+((-size.x-1)*(pos.y>0)*(pos.x>0))));
  diagonal.y = tex1Dfetch(texRef,(pixIdx+((-size.x+1)*(pos.y>0)*(pos.x<(size.x-1)))));
  diagonal.z = tex1Dfetch(texRef,(pixIdx+((size.x-1)*(pos.y<(size.y-1))*(pos.x>0))));
  diagonal.w = tex1Dfetch(texRef,(pixIdx+((size.x+1)*(pos.y<(size.y-1))*(pos.x<(size.x-1)))));
  cross.x = tex1Dfetch(texRef,(pixIdx-(size.x*(pos.y>0))));
  cross.y = tex1Dfetch(texRef,(pixIdx+(size.x*(pos.y<(size.y-1)))));
  cross.z = tex1Dfetch(texRef,(pixIdx-(pos.x>0)));
  cross.w = tex1Dfetch(texRef,(pixIdx+(pos.x<(size.x-1))));

  Lx = (-0.5*cross.z) + (0.5*cross.w);
  Ly = (0.5*cross.x) - (0.5*cross.y);
  Lxx = cross.z - (2.0*pixel) + cross.w;
  Lxy = (-0.25*diagonal.x) + (0.25*diagonal.y) + (0.25*diagonal.z) + (-0.25*diagonal.w);
  Lyy = cross.x -(2.0*pixel) + cross.y;


  Lvv[pixIdx] = (((Lx*Lx)*Lxx) + (2.0*Lx*Ly*Lxy) + (Ly*Ly*Lyy))/((Lx*Lx) + (Ly*Ly));

}

float* CudaCannyEdgeDetection::cuda2ndDerivative(dim3 DimGrid, dim3 DimBlock, const float *d_input, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  cudaFuncSetCacheConfig(kernel_Compute2ndDerivative,cudaFuncCachePreferNone);

 /// Se reserva memoria.
  float * d_Lvv;
  cudaMalloc((void**) &d_Lvv, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("La reserva de memoria para la imagen ha fallado.");

  /// Asociación de la textura al array.
  cudaBindTexture (NULL ,texRef, d_input);
  CUT_CHECK_ERROR("Texture bind failed");

  /// Configuración de la textura de lado del host.
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_Compute2ndDerivative<<<DimGrid,DimBlock>>>(d_Lvv, size);

  /// uDesasociación de la textura al array.
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("La liberación de memoria ha fallado.");

  return(d_Lvv);

}


__global__ void hysteresisPreparation_kernel(float *hysteresis, int3 size, float t1, float t2){

  ///ïndice de pixel a procesar.
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  float pixel;


  ///Índice de pixel de salida.
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(pixIdx < size.z){

    pixel = tex1Dfetch(mag_texRef, pixIdx)*tex1Dfetch(texRef,pixIdx);

    hysteresis[pixIdx] = ((POSSIBLE_EDGE-1)*(pixel>t2)+POSSIBLE_EDGE)*(pixel>t1);

  }

}

__global__ void hysteresisWrite_kernel(float *output, int3 size){

  ///Índice de pixel a tratar
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  float pixel;

  if(pixIdx < size.z){

    pixel = tex1Dfetch(hysTexRef, pixIdx);
    output[pixIdx] = (pixel==DEFINITIVE_EDGE) * pixel;

  }

}

__global__ void kernel_hysteresis_glm1D(float *hys_img, int3 size, int *have_modified_pixels){

  __shared__ float s_slice[SLICE_WIDTH*SLICE_WIDTH];
  __shared__ int modified_block_pixels; /// control del bucle interior.
  __shared__ int modified_image_pixels; /// control del bucle exterior
  int edge;

  // Píxel a tratar en el bloque.
  int2 slice_pos;
  slice_pos.y = threadIdx.x / SLICE_BLOCK_WIDTH;
  slice_pos.x = threadIdx.x - (slice_pos.y * SLICE_BLOCK_WIDTH);

  // índice del píxel en la imagen.
  int2 pos;
  pos.x = (slice_pos.x + (blockIdx.x * SLICE_BLOCK_WIDTH)) % size.x;
  pos.y = (((((blockIdx.x * SLICE_BLOCK_WIDTH))) / size.x) * SLICE_BLOCK_WIDTH ) + slice_pos.y;

  int sliceIdx = (slice_pos.y*SLICE_WIDTH) + slice_pos.x  + SLICE_WIDTH + 1;

  // índice del píxel en la imagen resultado de la histeresis.
  int pixIdx = pos.y * size.x + pos.x;

  if (!threadIdx.x) modified_image_pixels = NOT_MODIFIED;
  if (!pixIdx) have_modified_pixels[0] = 0;


  // se carga el centro
  if ((pos.x>0)&&(pos.y>0)&&(pos.x<(size.x-1))&&(pos.y<(size.y-1))){

    s_slice[sliceIdx] = hys_img[pixIdx];

    /// se carga la zona superior.
    if(!slice_pos.y){
      if(!slice_pos.x){
        s_slice[0] = hys_img[pixIdx-size.x-1];///<superior izquierda
      }
      s_slice[slice_pos.x+1] = hys_img[pixIdx-size.x];
      if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
        s_slice[SLICE_WIDTH-1] = hys_img[pixIdx-size.x+1];///<superior derecha
      }
    }
    /// se carga la parte inferior
    if(slice_pos.y == (SLICE_BLOCK_WIDTH-1)){
      if(!slice_pos.x){
        s_slice[SLICE_WIDTH*(SLICE_WIDTH-1)] = hys_img[pixIdx+size.x-1];///<BL
      }
      s_slice[(SLICE_WIDTH*(SLICE_WIDTH-1))+1+slice_pos.x] = hys_img[pixIdx+size.x];
      if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
        s_slice[(SLICE_WIDTH*SLICE_WIDTH)-1] = hys_img[pixIdx+size.x+1];///<BR
      }
    }
    /// se carga la parte izquierda
    if(!slice_pos.x){
      s_slice[(slice_pos.y+1)*SLICE_WIDTH] = hys_img[pixIdx-1];
    }
    /// se carga la aprte derecha
    if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
      s_slice[((slice_pos.y+2)*SLICE_WIDTH)-1] = hys_img[pixIdx+1];
    }

  }
  else{
    s_slice[sliceIdx] = 0;

    /// se carga la parte superior
    if(!slice_pos.y){
      if(!slice_pos.x){
        s_slice[0] = 0;///<superior izquierda
      }
      s_slice[slice_pos.x+1] = 0;
      if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
        s_slice[SLICE_WIDTH-1] = 0;///<superior derecha
      }
    }
    /// se carga la parte inferior
    if(slice_pos.y == (SLICE_BLOCK_WIDTH-1)){
      if(!slice_pos.x){
        s_slice[SLICE_WIDTH*(SLICE_WIDTH-1)] = 0;///<inferior izquierda
      }
      s_slice[(SLICE_WIDTH*(SLICE_WIDTH-1))+1+slice_pos.x] = 0;
      if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
        s_slice[(SLICE_WIDTH*SLICE_WIDTH)-1] = 0;///<inferior derecha
      }
    }
    /// se carga la aprte izquierda
    if(!slice_pos.x){
      s_slice[(slice_pos.y+1)*SLICE_WIDTH] = 0;
    }
    /// se carga la aprte derecha
    if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
      s_slice[((slice_pos.y+2)*SLICE_WIDTH)-1] = 0;
    }
  }

  __syncthreads();

  do{

    if (!threadIdx.x) modified_block_pixels = NOT_MODIFIED;

    __syncthreads();

    if(s_slice[sliceIdx] == POSSIBLE_EDGE){

    	// si la variable edge vale 1 al menos un vecino es una arista definitiva.
    	//si vale 0 es que no.
      edge = (!((s_slice[sliceIdx-SLICE_WIDTH-1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx-SLICE_WIDTH] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx-SLICE_WIDTH+1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx-1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+SLICE_WIDTH-1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+SLICE_WIDTH] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+SLICE_WIDTH+1] != DEFINITIVE_EDGE)));
      if ( (edge)*(!modified_block_pixels) ) modified_image_pixels = modified_block_pixels = MODIFIED;
      s_slice[sliceIdx] = POSSIBLE_EDGE + ((edge)*(POSSIBLE_EDGE-1));

    }

    __syncthreads();


  }while(modified_block_pixels);// final del bucle interior.

  // actualizar solamente los bloques que han modificado pixeles.
  if ((modified_image_pixels) && (pos.x < (size.x)) && (pos.y < (size.y))){
    hys_img[pixIdx] = s_slice[sliceIdx];
  }
  if (!threadIdx.x) have_modified_pixels[0] += modified_image_pixels;

}

float* CudaCannyEdgeDetection::cudaHysteresis(dim3 DimGrid, dim3 DimBlock, float *d_img, float *d_mag, int width, int height, float t1, float t2){
	//invocación del kernel del lado del host.

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  cudaFuncSetCacheConfig(hysteresisPreparation_kernel,cudaFuncCachePreferNone);
  cudaFuncSetCacheConfig(kernel_hysteresis_glm1D,cudaFuncCachePreferNone);
  cudaFuncSetCacheConfig(hysteresisWrite_kernel,cudaFuncCachePreferNone);

  float *d_hys;
  cudaMalloc((void**) &d_hys, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("la creación de memoria para la imagen resultado de histeresis ha fallado");

  /// asociación de textura con imagen
  cudaBindTexture (NULL ,texRef, d_img);
  cudaBindTexture (NULL ,mag_texRef, d_mag);
  CUT_CHECK_ERROR("Texture bind failed");

  /// se configura la textura.
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  hysteresisPreparation_kernel<<<DimGrid,DimBlock>>>(d_hys, size, t1, t2);

  /// se desasocian las texturas liberando la memoria.
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(mag_texRef);
  CUT_CHECK_ERROR("la liberación de memoria ha fallado.");

  // realización de la histéresis.
  int *modif;
  cudaMalloc((void**) &modif, (sizeof(int)));

  int cont[1];
  do{
    kernel_hysteresis_glm1D<<<DimGrid,DimBlock>>>(d_hys, size, modif);
    CUT_CHECK_ERROR("Hysteresis Kernel failed");
    cudaMemcpy(cont,modif,sizeof(int),cudaMemcpyDeviceToHost);
  }while(cont[0]);

  /// se asocia la textura al array cuda
  cudaBindTexture (NULL ,hysTexRef, d_hys);
  CUT_CHECK_ERROR("Texture bind failed");

  /// se configura dicha textura
  hysTexRef.normalized = false;
  hysTexRef.filterMode = cudaFilterModePoint;

  hysteresisWrite_kernel<<<DimGrid,DimBlock>>>(d_img, size);
  CUT_CHECK_ERROR("la histeresis ha fallado.");

  /// se libera la memoria.
  cudaUnbindTexture(hysTexRef);
  CUT_CHECK_ERROR("la desasociación de la textura ha fallado.");

  cudaFree(d_hys);
  cudaFree(modif);
  CUT_CHECK_ERROR("Memory free failed");

  float *d_edges;
  d_edges = d_img;
  return(d_edges);
}
