/*
 * CudaLaplaceShared.cu
 *
 *  Created on: 07/01/2013
 *      Author: jose
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <QTime>
#include "./src/common/Controlador.h"
#include "CudaLaplaceShared.cuh"
//Funciones para generación del kernerl para difuminado gaussiano.
#include "./src/cuda/canny/CudaDiscreteGaussian.cuh"
//Algoritmos en GPU compartida
#include "./src/cuda/Cuda5StepConvolutionBusiness.cuh"
//Funciones para generación de convolución en dos dimensiones mediante convolución separable.
#include "./src/cuda/canny/Cuda2DSeparableConvolution.cuh"
// Textura para almacenamiento de la imagen
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

#define THREADS_PER_BLOCK_MAX 256

__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale )
{
	short Sum = (short) -ur -um -ur -ml  +8*mm -mr -ll -lm -lr;
    if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
    return (unsigned char) Sum;
}

__global__ void
SobelShared( uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
             short BlockWidth, short SharedPitch,
#endif
             short w, short h, float fScale )
{
    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
        LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
            (float) (u+4*ib-RADIUS+0), (float) (v-RADIUS) );
        LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
            (float) (u+4*ib-RADIUS+1), (float) (v-RADIUS) );
        LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
            (float) (u+4*ib-RADIUS+2), (float) (v-RADIUS) );
        LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
            (float) (u+4*ib-RADIUS+3), (float) (v-RADIUS) );
    }
    if ( threadIdx.y < RADIUS*2 ) {
        //
    	//copiar tiras de tamaño RADIUS*2 a la memoria compartida
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
        for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
            LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
                (float) (u+4*ib-RADIUS+0), (float) (v+blockDim.y-RADIUS) );
            LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
                (float) (u+4*ib-RADIUS+1), (float) (v+blockDim.y-RADIUS) );
            LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
                (float) (u+4*ib-RADIUS+2), (float) (v+blockDim.y-RADIUS) );
            LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
                (float) (u+4*ib-RADIUS+3), (float) (v+blockDim.y-RADIUS) );
        }
    }

    __syncthreads();

    u >>= 2;    // los indices son uchar desde este punto
    uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

        unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
        unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
        unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
        unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
        unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
        unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
        unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
        unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
        unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

        uchar4 out;

        out.x = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale );

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeSobel(pix01, pix02, pix00,
                             pix11, pix12, pix10,
                             pix21, pix22, pix20, fScale );

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeSobel( pix02, pix00, pix01,
                              pix12, pix10, pix11,
                              pix22, pix20, pix21, fScale );

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeSobel( pix00, pix01, pix02,
                              pix10, pix11, pix12,
                              pix20, pix21, pix22, fScale );
        if ( u+ib < w/4 && v < h ) {
            pSobel[u+ib] = out;
        }
    }

    __syncthreads();
}

__global__ void
SobelCopyImage( Pixel *pSobelOriginal, unsigned int Pitch,
                int w, int h, float fscale )
{
    unsigned char *pSobel =
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        pSobel[i] = min( max((tex2D( tex, (float) i, (float) blockIdx.x ) * fscale), 0.f), 255.f);
    }
}

__global__ void
SobelTex( Pixel *pSobelOriginal, unsigned int Pitch,
          int w, int h, float fScale )
{
    unsigned char *pSobel =
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        unsigned char pix00 = tex2D( tex, (float) i-1, (float) blockIdx.x-1 );
        unsigned char pix01 = tex2D( tex, (float) i+0, (float) blockIdx.x-1 );
        unsigned char pix02 = tex2D( tex, (float) i+1, (float) blockIdx.x-1 );
        unsigned char pix10 = tex2D( tex, (float) i-1, (float) blockIdx.x+0 );
        unsigned char pix11 = tex2D( tex, (float) i+0, (float) blockIdx.x+0 );
        unsigned char pix12 = tex2D( tex, (float) i+1, (float) blockIdx.x+0 );
        unsigned char pix20 = tex2D( tex, (float) i-1, (float) blockIdx.x+1 );
        unsigned char pix21 = tex2D( tex, (float) i+0, (float) blockIdx.x+1 );
        unsigned char pix22 = tex2D( tex, (float) i+1, (float) blockIdx.x+1 );
        pSobel[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale );
    }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
{
    cudaChannelFormatDesc desc;

    if (Bpp == 1) {
        desc = cudaCreateChannelDesc<unsigned char>();
    } else {
        desc = cudaCreateChannelDesc<uchar4>();
    }

    cudaMallocArray(&array, &desc, iw, ih);
    cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice);
}

extern "C" void deleteTexture(void)
{
    cudaFreeArray(array);
}


extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, float fScale)
{
    cudaBindTextureToArray(tex, array);

    switch ( mode ) {
        case  SOBELDISPLAY_IMAGE:
            SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale );
            break;
        case SOBELDISPLAY_SOBELTEX:
            SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale );
            break;
        case SOBELDISPLAY_SOBELSHARED:
        {
            dim3 threads(16,4);
#ifndef FIXED_BLOCKWIDTH
	          int BlockWidth = 80; // must be divisible by 16 for coalescing
#endif
        		dim3 blocks = dim3(iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                               ih/threads.y+(0!=ih%threads.y));
        		int SharedPitch = ~0x3f&(4*(BlockWidth+2*RADIUS)+0x3f);
        		int sharedMem = SharedPitch*(threads.y+2*RADIUS);

        		// el ancho debe ser divisible entre 4.
        		iw &= ~3;
        		SobelShared<<<blocks, threads, sharedMem>>>((uchar4 *) odata,
                                                        iw,
#ifndef FIXED_BLOCKWIDTH
                                                        BlockWidth, SharedPitch,
#endif
                                                		    iw, ih, fScale );
        }
        break;
    }

    cudaUnbindTexture(tex);
}


CudaLaplaceShared::CudaLaplaceShared() {
	// TODO Auto-generated constructor stub

}

CudaLaplaceShared::~CudaLaplaceShared() {
	// TODO Auto-generated destructor stub
}


float* CudaLaplaceShared::convolve(float *imagen, int ancho,
		int alto, int brillo) {
	Pixel *img;
	Pixel *h_img;
	int DATA_SIZE=0;
	DATA_SIZE = alto * ancho * sizeof(unsigned char);
	cudaMalloc((void **) &img, DATA_SIZE);
	h_img = (unsigned char*) malloc(DATA_SIZE);
	int i=0;
	int x=0;
	for(i=0;i<ancho*alto;i++){
		h_img[i] = (unsigned char)imagen[i];
	}

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

	setupTexture(ancho,alto,h_img,1);
	sobelFilter(img, ancho, alto, SOBELDISPLAY_SOBELSHARED, brillo );

	//Se copia el resultado de la ejecución de vuelta a CPU
	cudaMemcpy(h_img, img, DATA_SIZE, cudaMemcpyDeviceToHost);
	int elapsed =time->elapsed();
	Controlador *controlador = Controlador::Instance();
	controlador->setGpuExecutionTime(elapsed);

	float *h_result;
	h_result = (float*) malloc( alto * ancho * sizeof(float));
	for(i=0;i<ancho*alto;i++){
		h_result[i] = (float)h_img[i];
		}
	return h_result;

}



float* CudaLaplaceShared::laplacianOfGaussianShared(float *imagen, int ancho,int alto, int brillo,int radius,float sigma){
		Pixel *img;
		dim3 dimgrid(((radius*2)+1),1,1);
		dim3 dimblock(((radius*2)+1),1,1);

		//Puntero a float para kernel de difuminado de gauss en device.
		float *d_gauss_kernel;
		//Puntero a float para kernel de difuminado de gauss en host.
		float *retorno;
		//Puntero a float para matriz de imagen en device.
		float *d_img;
		float *d_img_blurred;
		//Puntero a float en host de la imagen difuminada.
		float *h_img_blurred;

		//Valor de la 2a derivada en cada punto de la imagen
		float *deriv;
		//Valor de la posición en cada punto de la derivada de la imagen.
		float *mag;
		//resultado de la supresio de no-máximos
		float *nomaxi;
		//Resultado del proceso de canny en device.
		float *d_result_canny;

		//******TIMER
		QTime *time = new QTime();
		time->start();
		//********
		//1 - Primer paso construir el kernel para gauss con cuda.-----------------------------
		CudaDiscreteGaussian gaussian = CudaDiscreteGaussian();
		d_gauss_kernel = gaussian.cuda1DGaussianOperator(dimgrid,dimblock,((radius*2)+1),sigma);

		dim3 dimgridGauss(THREADS_PER_BLOCK_MAX,1,1);
		dim3 dimblockGauss((((alto*ancho))+THREADS_PER_BLOCK_MAX-1)/THREADS_PER_BLOCK_MAX,1,1);

		//2.0 Copiado de la imagen de puntero de host a puntero de device.
		//Reservamos memoria en device
		cudaMalloc((void **) &d_img, (alto*ancho)*sizeof(float));
		//Realizamos el copiado de los datos.
		cudaMemcpy(d_img, imagen, ((alto*ancho)*sizeof(float)),cudaMemcpyHostToDevice);

		//2 - Paso 2 : realizar la convolución del kernel creado para prcesar el difuminado gaussiano sobre la imagen.
		//2.1 - Instanciamos el objeto que proporciona las funcionalidades de convolución en 2D
		Cuda2DSeparableConvolution convolve = Cuda2DSeparableConvolution();
		//2.2 - Informamos los argumentos necesarios para realizar la convolución.
		d_img_blurred = convolve.cuda2DSeparableConvolution(dimblockGauss,dimgridGauss,d_img,ancho,alto,d_gauss_kernel,((radius*2)+1),d_gauss_kernel,((radius*2)+1));
		//2.3 - Copiamos la información de la imagen difuminada de vuelta al host.
		int  timePassed =time->elapsed();
		h_img_blurred = (float*) malloc(((alto*ancho)*sizeof(float)));
		cudaMemcpy(h_img_blurred,d_img_blurred, ((alto*ancho)*sizeof(float)), cudaMemcpyDeviceToHost);

		Pixel *h_img;
		int DATA_SIZE=0;
		DATA_SIZE = alto * ancho * sizeof(unsigned char);
		cudaMalloc((void **) &img, DATA_SIZE);
		//cudaMalloc((void **) &img, DATA_SIZE);
		h_img = (unsigned char*) malloc(DATA_SIZE);

			float *intermedia1;
			intermedia1 = (float*) malloc(
					ancho * alto);
			int i=0;
		for(i=0;i<ancho*alto;i++){
				float a= h_img_blurred[i];
				if(a>255){a=255;}else if(a<0){a=0;}
				h_img[i] = (unsigned char)a;
			}

		//******TIMER
			QTime *time2 = new QTime();
			time2->start();
			//********
		//cudaMemcpy(img, h_img, DATA_SIZE, cudaMemcpyHostToDevice);
			setupTexture(ancho,alto,h_img,1);
			sobelFilter(img, ancho, alto, SOBELDISPLAY_SOBELSHARED, brillo );
			int timer2 =time2->elapsed();
			Controlador *controlador = Controlador::Instance();
			controlador->setGpuExecutionTime(timePassed+timer2);
			//Se copia el resultado de la ejecución de vuelta a CPU
			//cudaMemcpy(img, h_img, DATA_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpy(h_img, img, DATA_SIZE, cudaMemcpyDeviceToHost);

			float *h_result;
			h_result = (float*) malloc( alto * ancho * sizeof(float));

			for(i=0;i<ancho*alto;i++){
				h_result[i] = (float)h_img[i];
				}
			return h_result;
			//Se copia de vuelta la imagen procesada a CPU

		return h_img_blurred;
}
