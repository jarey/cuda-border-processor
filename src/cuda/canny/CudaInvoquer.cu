/*
 * CudaInvoquer.cu
 *
 * Creado: 05/12/2012
 * Autor: jose
 * Descripción: clase encargada de realizar el proceso de canny invocando a todos los procesos
 * intermedios necesarios:
 * 	1- generación del filtro gaussiano.
 * 	2- difuminación de la imagen aplicando el filtro de gauss creado.
 * 	3- cálculo del gradiente de la imagen en cada punto.
 * 	4- cálculo del cruce en 0.
 * 	5- procesamiento de histéresis para confeccionar los bordes finales.
 */
#include <iostream>
#include <stdio.h>
#include <cutil.h>
#include "CudaInvoquer.cuh"
#include "CudaCannyEdgeDetection.cuh"
//Funciones para generación del kernerl para difuminado gaussiano.
#include "CudaDiscreteGaussian.cuh"
//Funciones para generación de convolución en dos dimensiones mediante convolución separable.
#include "Cuda2DSeparableConvolution.cuh"
//Funciones relativas a canny (obtención de segunda derivada, obtención de dirección, supresión de no-maximos y hysteresis).
#include "CudaCannyEdgeDetection.cuh"
//Funciones de sobel y de supresión de no-maximos
#include "CudaZeroCrossing.cuh"
#include <QTime>
#include "./src/common/Controlador.h"


#define THREADS_PER_BLOCK 256

CudaInvoquer::CudaInvoquer() {
	// TODO Auto-generated constructor stub

}

CudaInvoquer::~CudaInvoquer() {
	// TODO Auto-generated destructor stub
}


float* CudaInvoquer::invoque(float* imagenEntrada,int imagen_width,int imagen_height,int radius, float sigma,float lowerThreshold, float upperThreshold,bool histeresis){

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

	//1 - Primer paso construir el kernel para gauss con cuda.-----------------------------
	CudaDiscreteGaussian gaussian = CudaDiscreteGaussian();
	d_gauss_kernel = gaussian.cuda1DGaussianOperator(dimgrid,dimblock,((radius*2)+1),sigma);

	dim3 dimgridGauss(THREADS_PER_BLOCK,1,1);
	dim3 dimblockGauss((((imagen_height*imagen_width))+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,1,1);

	//printf("El TAMAÑO de bloque es : %d \n", (((imagen_height*imagen_width))+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

	//2.0 Copiado de la imagen de puntero de host a puntero de device.
	//Reservamos memoria en device
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_img, (imagen_height*imagen_width)*sizeof(float)));
	//Realizamos el copiado de los datos.
	CUDA_SAFE_CALL(cudaMemcpy(d_img, imagenEntrada, ((imagen_height*imagen_width)*sizeof(float)),cudaMemcpyHostToDevice));

	//Comienzo de medición
	Controlador *controlador = Controlador::Instance();
	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	//2 - Paso 2 : realizar la convolución del kernel creado para prcesar el difuminado gaussiano sobre la imagen.
	//2.1 - Instanciamos el objeto que proporciona las funcionalidades de convolución en 2D
	Cuda2DSeparableConvolution convolve = Cuda2DSeparableConvolution();
	//2.2 - Informamos los argumentos necesarios para realizar la convolución.
	d_img_blurred = convolve.cuda2DSeparableConvolution(dimblockGauss,dimgridGauss,d_img,imagen_width,imagen_height,d_gauss_kernel,((radius*2)+1),d_gauss_kernel,((radius*2)+1));

	//3- Tras difuminar la imagen se debe calcular la aproximación de la segunda derivada de la imagen, para ello
	//se aplica el algoritmo de sobel hayando la magnitud de gradiente en cada punto y la dirección del mismo.
	CudaCannyEdgeDetection cannyObject = CudaCannyEdgeDetection();
	//3.1 obtención de la magnitud
	deriv = cannyObject.cuda2ndDerivative(dimblockGauss,dimgridGauss,d_img_blurred,imagen_width,imagen_height);
	//3.3 obtención de la dirección
	mag = cannyObject.cuda2ndDerivativePos(dimblockGauss,dimgridGauss,d_img_blurred,deriv,imagen_width,imagen_height);

	//4- Tras obtener la magnitud en cada punto y su dirección se debe realiza la supresión de no-máximos.
	CudaZeroCrossing zeroCrossing = CudaZeroCrossing();

	nomaxi = zeroCrossing.cudaZeroCrossing(dimblockGauss,dimgridGauss,deriv,imagen_width,imagen_height);

	if(histeresis){
	//5- Tras realizar la supresión de no-máximos el último paso es realizar la hystéresis.
	d_result_canny = cannyObject.cudaHysteresis(dimblockGauss,dimgridGauss,nomaxi,mag,imagen_width,imagen_height,lowerThreshold,upperThreshold);
	}else{
		d_result_canny = nomaxi;
	}
	//Final de medición
	int  timePassed =time->elapsed();
	controlador->setGpuExecutionTime(timePassed);

	h_img_blurred = (float*) malloc(((imagen_height*imagen_width)*sizeof(float)));
	cudaMemcpy(h_img_blurred,d_result_canny, ((imagen_height*imagen_width)*sizeof(float)), cudaMemcpyDeviceToHost);

	cudaFree(deriv);
	cudaFree(mag);
	cudaFree(nomaxi);

	return h_img_blurred;
}
