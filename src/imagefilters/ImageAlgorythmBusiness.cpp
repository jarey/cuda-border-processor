/*
 * ImageAlgorythmBusiness.cpp
 *
 * Creado: 24/03/2012
 * Autor: jose
 * Descripción: clase encargada de gestionar toda la lógica de negocio de la aplicación en cuanto a la
 * ejecución de filtros de imágenes se refiere. En esta clase se implementa el desarrollo de la totalidad de
 * algoritmos de procesamiento de imágenes del proyecto y sus respectivas llamadas a clases externas.
 */

#include "ImageAlgorythmBusiness.h"
#include "./src/cuda/CudaAlgorythmBusiness.cuh"
#include "./src/cuda/Cuda2dConvolutionBusiness.cuh"
#include <iostream>
#include <stdio.h>
#include <QImage>
#include <math.h>
#include "./src/common/Constants.h"
#include "./src/common/Controlador.h"
#include "./src/canny/CImage.h"
#include "./src/cuda/CudaConvolutionSeparableBusiness.cuh"
//Include para emplear los timers de QT
#include <QTime>
#include "./src/common/TimeContainer.h"
//Algoritmos en GPU compartida
#include "./src/cuda/Cuda5StepConvolutionBusiness.cuh"
#include "./src/cuda/Cuda5bStepConvolutionBusiness.cuh"
#include "./src/cuda/canny/CudaSobelEdgeDetection.cuh"
#include "./src/cuda/canny/CudaInvoquer.cuh"
#include "./src/cuda/CudaRobertCrossShared.cuh"
#include "./src/cuda/CudaLaplaceShared.cuh"

ImageAlgorythmBusiness::ImageAlgorythmBusiness() {
	// TODO Auto-generated constructor stub

}

ImageAlgorythmBusiness::~ImageAlgorythmBusiness() {
	// TODO Auto-generated destructor stub
}

//***************************************************************
//						MÉTODOS AUXILIARES
//***************************************************************

/**
 * Combierte una imagen encapsulada en el tipo de dato QImage de QT a
 * puntero de floats en blanco y negro.
 */
float* ImageAlgorythmBusiness::fromQImageToMatrix(QImage imagenEntrada) {
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	float *matrix;
	//Variable para un pixel de la imagen.
	QRgb pixel;
	//Variable para el pintado de la imagen final.
	QColor mycolor;
	//--------------------------------------
	//Se reserva memoria para la matrix de aproximación de gradiente de la imagen.
	matrix = (float*) malloc(srcHeight * srcWidth * sizeof(float));

	float max = 0;
	int gray = 0;
	//Se itera sobre los pixeles de la imagen obteniendo una aproximación
	//del gradiente en cada punto por convolución de los filtros.

	for (int f = 0; f < srcWidth; f++) {
		for (int g = 0; g < srcHeight; g++) {
			pixel = imagenEntrada.pixel(f, g);

			gray = (0.299 * qRed(pixel) + .587 * qGreen(pixel) + .114 * qBlue(
					pixel));
			matrix[f + g * srcWidth] = gray;
		}
	}

	return matrix;
}

float* ImageAlgorythmBusiness::fromQImageToMatrix(QImage imagenEntrada, float *puntero) {
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	//Variable para un pixel de la imagen.
	QRgb pixel;
	//Variable para el pintado de la imagen final.
	QColor mycolor;

	//Se reserva memoria para la matrix de aproximación de gradiente de la imagen.
	puntero = (float*) malloc(srcHeight * srcWidth * sizeof(float));
	float max = 0;
	int gray = 0;
	//Se itera sobre los pixeles de la imagen obteniendo una aproximación
	//del gradiente en cada punto por convolución de los filtros.

	for (int f = 0; f < srcWidth; f++) {
		for (int g = 0; g < srcHeight; g++) {
			pixel = imagenEntrada.pixel(f, g);

			gray = (0.299 * qRed(pixel) + .587 * qGreen(pixel) + .114 * qBlue(
					pixel));
			puntero[f + g * srcWidth] = gray;
		}
	}

	return puntero;
}

/**
 * Combierte un puntero a float que representa una imagen en blanco y negro a
 * una imagen encapsulada en el tipo de dato QImage de QT.
 */
QImage ImageAlgorythmBusiness::fromMatrixToQImage(float* matrix, int width,
		int height) {
	//Variables
	//Variable para la imagen intermedia resultado del preprocesado mediante el filtro de gauss
	QImage gauss;
	QImage grayImage;
	float *gradient;
	float *orientation;
	float *nonmax;
	int srcHeight;
	int srcWidth;
	QRgb pixel;
	//Variable para el pintado de la imagen final.
	QColor mycolor;
	float max = 0;

	gauss = QImage(width, height, QImage::Format_RGB32);
	//Las variables de ancho y alto toman el valor del ancho y el alto de la imagen de entrada.
	srcHeight = height;
	srcWidth = width;

	for (int i = 0; i < srcWidth; i++) {
		for (int j = 0; j < srcHeight; j++) { //Se inicializan las variables para el cálculo de l
			max = matrix[i + j * srcWidth] > max ? matrix[i + j * srcWidth]
					: max;
		}
	}

	for (int i = 0; i < srcWidth; i++) {
		for (int j = 0; j < srcHeight; j++) {
			//printf("valor de la iteracción actual: ");
			//printf("%f\n",matrix[i + j * srcWidth] / max);
			float a =(matrix[i + j * srcWidth] / max);
			if(a>1){a=1;}else if(a<0){a=0;}
			mycolor.setHsv(0, 0,
					(int(255.0 * /*(matrix[i + j * srcWidth] / max)*/a)));
			gauss.setPixel(i, j, mycolor.rgb());
		}
	}

	free(gradient);
	free(orientation);
	free(nonmax);

	return gauss;
}

/**
 * Combierte una imagen encapsulada en el tipo de dato QImage de QT a
 * escala de grises.
 */
QImage ImageAlgorythmBusiness::grayScale(QImage imagenEntrada, int pad) {
	//Variables
	//Ancho y alto de la imagen fuente.
	int srcWidth;
	int srcHeight;
	//Ancho y alto ampliados para la imagen en blanco y negro.
	int grayWidth;
	int grayHeight;
	//Variable para el valor de gris.
	double gray;
	//Variables de iteracción en el bucle.
	int i, j;
	//Variable para la imagen en blanco y negro.
	QImage grayImg;
	//Variable para un pixel de la imagen.
	QRgb pixel;

	//Ancho y alto de la imagen fuente.
	srcWidth = imagenEntrada.width();
	srcHeight = imagenEntrada.height();
	//Se amplia la imagen en blanco y negro para no salirse de los límites de la imagen.
	grayWidth = srcWidth + pad;
	grayHeight = srcHeight + pad;
	//Imagen en blanco y negro a devolver.
	grayImg = QImage(grayWidth, grayHeight, QImage::Format_RGB32);

	//Para cada uno de los píxeles de la imagen original se obtiene dicho pixel
	//y se convierte a blanco y negro.
	//Tras ello se setea en las coordenadas i,j + el padding de la imagen resultado.
	for (i = 0; i < srcWidth; i++) {
		for (j = 0; j < srcHeight; j++) {
			pixel = imagenEntrada.pixel(i, j);

			gray = (0.299 * qRed(pixel) + .587 * qGreen(pixel) + .114 * qBlue(
					pixel));
			grayImg.setPixel(i + pad / 2, j + pad / 2,
					qRgb(int(gray), int(gray), int(gray)));
		}
	}

	//Se replican los extremos para limite superior,inferior, izquierdo,derecho y esquinas.
	//ya que la imagen resultado será PAD pixeles más grande que la imagen original.
	for (int i = (pad / 2); i < grayWidth - (pad / 2); i++) {
		for (int a = 0; a < (pad / 2); a++) {
			grayImg.setPixel(i, a, grayImg.pixel(i, (pad / 2)));
			grayImg.setPixel(i, (a + grayHeight) - (pad / 2),
					grayImg.pixel(i, grayHeight - (pad / 2) - 1));
		}
	}

	for (int i = (pad / 2); i < grayHeight - (pad / 2); i++) {
		for (int a = 0; a < (pad / 2); a++) {
			grayImg.setPixel(a, i, grayImg.pixel(i, (pad / 2) - 1));
			grayImg.setPixel((a + grayWidth) - (pad / 2), i,
					grayImg.pixel(grayWidth - (pad / 2) - 1, i));
		}
	}
	for (int a = 0; a < (pad / 2); a++) {
		for (int b = 0; b < (pad / 2); b++) {
			grayImg.setPixel(a, b, grayImg.pixel((pad / 2), (pad / 2)));
			grayImg.setPixel(a + grayWidth - (pad / 2), b,
					grayImg.pixel(grayWidth - (pad / 2), (pad / 2)));
			grayImg.setPixel(a, b + grayHeight - (pad / 2),
					grayImg.pixel((pad / 2), grayHeight - (pad / 2)));
			grayImg.setPixel(
					a + grayWidth - (pad / 2),
					b + grayHeight - (pad / 2),
					grayImg.pixel(grayWidth - (pad / 2), grayHeight - (pad / 2)));
		}
	}
	//Se retorna la imagen resultado de la aplicación del filtro.
	return grayImg;
}

/**
 * Genera los kernels de gauss para unos valores de datos de sigma y radio.
 */
float* ImageAlgorythmBusiness::generateGaussianKernels(float sigma, int gaussRadius) {
//Consideraciones : kernelX y kernelY deben llegar como punteros con memoria reservada para
//GAUSS_WIN_WIDTH sino se producirá un fallo.
	int GAUSS_WIN_RADI = gaussRadius;
	int GAUSS_WIN_WIDTH = GAUSS_WIN_RADI * 2 + 1;

	float *kernelX = (float*) malloc(GAUSS_WIN_WIDTH * sizeof(float));
	float *kernelY = (float*) malloc(GAUSS_WIN_WIDTH * sizeof(float));

	float xK[GAUSS_WIN_WIDTH], yK[GAUSS_WIN_WIDTH];
	float sumX = 0, sumY = 0;
	for (int a = -GAUSS_WIN_RADI; a <= GAUSS_WIN_RADI; ++a) {
		yK[a + GAUSS_WIN_RADI] = exp(-a * a / (2 * sigma * sigma));
		xK[a + GAUSS_WIN_RADI] = (1.0 / (6.283185307179586 * sigma * sigma))
				* yK[a + GAUSS_WIN_RADI];

		sumX += xK[a + GAUSS_WIN_RADI];
		sumY += yK[a + GAUSS_WIN_RADI];
	}

	//Normalize
	for (int a = 0; a < GAUSS_WIN_WIDTH; ++a) {
		xK[a] /= sumX;
		kernelX[a]=xK[a];
		printf("%f ", xK[a]);
	}
	printf("\n");
	for (int a = 0; a < GAUSS_WIN_WIDTH; ++a) {
		yK[a] /= sumY;
		kernelY[a]=yK[a];
		printf("%f ", yK[a]);
	}

	return kernelX;
}

/**
 * Aplica el algoritmo de gauss a una imagen de entrada, aplicando el filtro de gauss
 * generado para los parámetros de radio y sigma indicados.
 */
QImage ImageAlgorythmBusiness::Gauss(QImage imagenEntrada, int radioGauss,
		double sigmaGauss) {
	//Variables:
	//Variables para el ancho y alto de la imagen original
	int srcWidth;
	int srcHeight;
	//variable para la imagen resultado de la aplicación del filtro.
	QImage result;
	//Variable para la imagen resultado del paso a escala de grises de la imagen original.
	QImage grayImg;
	//Puntero a float para el kernel de tamaño variable creado de forma dinámica.
	float *kernel;
	//Variable para almacenar el tamaño del kernnel.
	int dim;
	//Variable para almacenar la mitad entera del kernnel.
	int half;
	//Variable para almacenar el máximo valor para e kernnel (posterior normalización)
	float total;
	//Variable para un pixel de la imagen.
	QRgb pixel;
	//puntero a float para la matriz de aproximación del gradiente de la imagen en cada punto.
	float *Sobel_norm;
	//Variable para almacenar elmaximo del gradiente (posterior normalización)
	float max = 0.0;
	//Variable para almacenar el valor del gradiente en cada punto.
	double value = 0.0;
	//Variable para el color
	QColor mycolor;
	//----------------------------------------------
	//Se reserva memoria dinámicamente para el kernnel dependiendo del radio para el kernnel indicado en la interfaz.
	kernel = (float*) malloc(radioGauss * radioGauss * sizeof(float));
	//Se calcula la dimensión como el cuadrado del radio indicado.
	dim = radioGauss * radioGauss;

	//Se calcula la mitad como la división entera del radio indicado.
	half = radioGauss / 2; // división entera (5/2=2)
	total = 0;

	// Cálculo coeficientes
	for (int y = -half; y <= half; y++) {
		for (int x = -half; x <= half; x++) {
			int index = radioGauss * (y + half) + (x + half);
			kernel[index] = exp(-(x * x + y * y) / (sigmaGauss * sigmaGauss))
					/ 2 * 3.14 * sigmaGauss * sigmaGauss;
			total += kernel[index];
		}
	}
	// Normalización
	for (int i = 0; i < dim; i++) {
		kernel[i] /= total;
	}

	//El ancho y alto se toman de la imagen original.
	srcWidth = imagenEntrada.width();
	srcHeight = imagenEntrada.height();
	//Se reserva memoria por dicho alto y ancho para la matriz de aproximación del gradiente.
	Sobel_norm = (float*) malloc(srcHeight * srcWidth * sizeof(float));
	//Se pasa la imagen a blanco y negro.
	grayImg = grayScale(imagenEntrada, half * 2);

	//La imagen resultado toma el ancho,alto y formato de la imagen original.
	result = QImage(srcWidth, srcHeight, QImage::Format_RGB32);

	//Una vez pasada a grises se le aplica el filtro de gauss para limpiar el ruido.
	max = 0.0;
	value = 0.0;

	//Se recorre la imagen a lo alto
	for (int i = 0; i < srcWidth; i++) {
		for (int j = 0; j < srcHeight; j++) {

			value = 0.0;

			for (int k = 0; k < radioGauss; k++) {
				for (int l = 0; l < radioGauss; l++) {
					pixel = grayImg.pixel((i + half) + (half - k),
							(j + half) + (half - l));
					value = value + kernel[l * radioGauss + k] * qRed(pixel);
				}
			}
			//Se almacena el valor calculado.
			Sobel_norm[i + j * srcWidth] = value;
			//Se mantiene el máximo para posterior normalización
			max = Sobel_norm[i + j * srcWidth] > max ? Sobel_norm[i + j
					* srcWidth] : max;
		}
	}

	//Se pinta la imagen resultado de la aplicación del filtro.
	for (int i = 0; i < srcWidth; i++) {
		for (int j = 0; j < srcHeight; j++) {
			mycolor.setHsv(0, 0,
					int(255.0 * Sobel_norm[i + j * srcWidth]) / max);
			result.setPixel(i, j, mycolor.rgb());
		}
	}
	//Se retorna la imagen resultado de la aplicación del filtro.
	return result;
}

/**
 * Aplica la convolución CPU a una imagen, con el filtro indicado.
 */
QImage ImageAlgorythmBusiness::convolutionCPU(int sourceImageWidth,
		int sourceImageHeight, QImage paddedImage, ImageFilter imageFilter) {

	//****TIMER
	QTime *timer = new QTime();
	timer->start();
	//****
	Constants* constants = Constants::Instance();
	//Imagen para el resultado del algoritmo
	QImage result;
	//Variable para el máximo de la matriz de la imagen.
	float max = 0.0;
	//Variables para la aproximación del gradiente en el eje x e y.
	double value_gx, value_gy;
	//Matriz de resultados para la aproximación del gradiente.
	float *Sobel_norm;
	//Variable para un pixel de la imagen.
	QRgb pixel;
	//Variable para el pintado de la imagen final.
	QColor mycolor;
	//Radio del filtro
	int filterRadius = imageFilter.getFilterMatrixX().getWidth() - 1;
	//Se reserva memoria para la matrix de aproximación de gradiente de la imagen.
	Sobel_norm = (float*) malloc(
			sourceImageHeight * sourceImageWidth * sizeof(float));

	//La imagen resultado se instancia con el ancho y el alto de la imagen original asi como con su formato.
	result = QImage(sourceImageWidth, sourceImageHeight, paddedImage.format());
	if (imageFilter.getFilterMatrixX().getElements() != NULL
			&& imageFilter.getFilterMatrixY().getElements() != NULL) {

		//Convolución sobre un filtro bidimensional
		//Se itera sobre los pixeles de la imagen obteniendo una aproximación
		//del gradiente en cada punto por convolución de los filtros.
		for (int i = 0; i < sourceImageWidth; i++) {
			for (int j = 0; j < sourceImageHeight; j++) { //Se inicializan las variables para el cálculo de la aproximación del
				//gradiente en el punto.
				value_gx = 0.0;
				value_gy = 0.0;

				for (int k = 0; k < imageFilter.getFilterMatrixX().getWidth(); k++) {
					for (int l = 0; l
							< imageFilter.getFilterMatrixX().getWidth(); l++) {
						// Con el padding ya no hay problema para acceder a los píxeles de la imagen.
						//Se realiza la aproximación mediante circonvolución de los filtros en el
						//punto i,j.
						pixel = paddedImage.pixel(
								(i + (filterRadius / 2)) + ((filterRadius / 2)
										- k),
								(j + (filterRadius / 2)) + ((filterRadius / 2)
										- l));
						value_gx += imageFilter.getFilterMatrixX().getPos(l, k)
								* qRed(pixel);
						value_gy += imageFilter.getFilterMatrixY().getPos(l, k)
								* qRed(pixel);
					}
				}
				//Se obtiene la aproximación del gradiente en el punto i,j como la raiz cuadrada
				//de la suma de cuadrados de las aproximaciones en x e y.
				if (imageFilter.getFilterName()
						== constants->getSobelSquaredConstant()) {
					Sobel_norm[i + j * sourceImageWidth] = ((value_gx
							* value_gx) + (value_gy * value_gy)) / 8;
				} else {
					Sobel_norm[i + j * sourceImageWidth] = sqrt(
							value_gx * value_gx + value_gy * value_gy);
				}
				//Obtención del máximo para posterior nomralización de la matriz de valores.
				max = Sobel_norm[i + j * sourceImageWidth] > max ? Sobel_norm[i
						+ j * sourceImageWidth] : max;
			}
		}

	} else {
		//Convolución sobre un filtro unidimensional
		for (int i = 0; i < sourceImageWidth; i++) {
			for (int j = 0; j < sourceImageHeight; j++) { //Se inicializan las variables para el cálculo de la aproximación del
				//gradiente en el punto.
				value_gx = 0.0;
				value_gy = 0.0;

				for (int k = 0; k < imageFilter.getFilterMatrixX().getWidth(); k++) {
					for (int l = 0; l
							< imageFilter.getFilterMatrixX().getWidth(); l++) {
						// Con el padding ya no hay problema para acceder a los píxeles de la imagen.
						//Se realiza la aproximación mediante circonvolución de los filtros en el
						//punto i,j.
						pixel = paddedImage.pixel(
								(i + (filterRadius / 2)) + ((filterRadius / 2)
										- k),
								(j + (filterRadius / 2)) + ((filterRadius / 2)
										- l));
						value_gx += imageFilter.getFilterMatrixX().getPos(l, k)
								* qRed(pixel);
					}
				}
				//Se obtiene la aproximación del gradiente en el punto i,j como la raiz cuadrada
				//de la suma de cuadrados de las aproximaciones en x e y.
				//Se asegura que el valor del gradiente en el punto esté en el intervalo [0,255]
				if (value_gx > 255) {
					value_gx = 255;
				}
				if (value_gx < 0) {
					value_gx = 0;
				}
				//Se asigna el valor de la aproximación a la matrix de aproximación del gradiente en i,j.
				Sobel_norm[i + j * sourceImageWidth] = value_gx;
				// Captura del maximo para posterior normalización.
				max = Sobel_norm[i + j * sourceImageWidth] > max ? Sobel_norm[i
						+ j * sourceImageWidth] : max;
			}
		}
	}

	//Se recorre pixel a pixel la matriz de la aproximación del gradiente, pintando negro o blanco
	//en la matriz normalizada a la imagen final resultado.
	for (int i = 0; i < sourceImageWidth; i++) {
		for (int j = 0; j < sourceImageHeight; j++) {
			mycolor.setHsv(0, 0,
					int(255.0 * Sobel_norm[i + j * sourceImageWidth]) / max);
			result.setPixel(i, j, mycolor.rgb());
		}
	}
	//Se retorna la imagen final.
	delete Sobel_norm;

	return result;
}

//***************************************************************
//						EJECUCIÓN EN CPU
//***************************************************************

/**
 * Aplica el algoritmo de Sobel en CPU sobre una imagen.
 */
QImage ImageAlgorythmBusiness::SobelFilterCPU(QImage imagenEntrada,
		SobelImageFilter sobelFilter) {
	//*******TIMER
		QTime *time = new QTime();
		time->start();
	//*******

	//Variables
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	//Imagen resultado del paso a escala de grises.
	QImage grayImg;

	//Radio del filtro
	int filterRadius = sobelFilter.getFilterMatrixX().getWidth() - 1;

	//Se pasa la imagen original a escala de grises con 2 pixels de padding(filtro 3x3).
	grayImg = this->grayScale(imagenEntrada, filterRadius);

	//Se retorna la imagen final.
	QImage result;
	result= convolutionCPU(srcWidth, srcHeight, grayImg, sobelFilter);
	//*****TIMER
	double timePassed = time->elapsed();
	//*****
	manageExecutionTimeWriting(timePassed,"Sobel","CPU",true);

	return result;
}

/**
 * Aplica el algoritmo de Prewitt en CPU sobre una imagen.
 */
QImage ImageAlgorythmBusiness::PrewittFilterCPU(QImage imagenEntrada,
		PrewittImageFilter prewittFilter) {
	//*******TIMER
	QTime *time = new QTime();
	time->start();
	//*******
	//Variables
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	//Imagen resultado del paso a escala de grises.
	QImage grayImg;

	//Radio del filtro
	int filterRadius = prewittFilter.getFilterMatrixX().getWidth() - 1;

	//Se pasa la imagen original a escala de grises con 2 pixels de padding(filtro 3x3).
	grayImg = this->grayScale(imagenEntrada, filterRadius);

	//Se retorna la imagen final.
	QImage result;
	result = convolutionCPU(srcWidth, srcHeight, grayImg, prewittFilter);
	//*****
	int timePassed = time->elapsed();
	manageExecutionTimeWriting(timePassed,"Prewitt","CPU",true);
	//LLamada a función que gestiona la escritura de los valores de tiempo en el listado de cada funcionalidad.
	return result;
}

/**
 * Aplica el algoritmo de Laplace en CPU para una imagen.
 */
QImage ImageAlgorythmBusiness::LaplaceFilterCPU(QImage imagenEntrada,
		LaplaceImageFilter laplaceFilter) {
	//*******TIMER
	QTime *time = new QTime();
	time->start();
	//*******
	//Variables
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	//Imagen resultado del paso a escala de grises.
	QImage grayImg;
	//Radio del filtro
	int filterRadius = laplaceFilter.getFilterMatrixX().getWidth() - 1;
	//Se pasa la imagen original a escala de grises con 2 pixels de padding(filtro 3x3).
	grayImg = this->grayScale(imagenEntrada, filterRadius);
	//Se retorna la imagen final.
	QImage result;
	result = convolutionCPU(srcWidth, srcHeight, grayImg, laplaceFilter);
	int timePassed = time->elapsed();
	manageExecutionTimeWriting(timePassed,"Laplace","CPU",true);
	return result;
}

/**
 * Aplica el algoritmo de Robert Cross en CPU sobre una imagen.
 */
QImage ImageAlgorythmBusiness::RobertCrossFilterCPU(QImage imagenEntrada,
		RobertCrossImageFilter robertCrossFilter) {
	//*******TIMER
	QTime *time = new QTime();
	time->start();
	//*******
	//Variables
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	//Imagen resultado del paso a escala de grises.
	QImage grayImg;

	//Radio del filtro
	int filterRadius = robertCrossFilter.getFilterMatrixX().getWidth() - 1;

	//Se pasa la imagen original a escala de grises con 2 pixels de padding(filtro 3x3).
	grayImg = this->grayScale(imagenEntrada, filterRadius);

	//Se retorna la imagen final.
	QImage result;
	result = convolutionCPU(srcWidth, srcHeight, grayImg, robertCrossFilter);

	int timePassed = time->elapsed();
	manageExecutionTimeWriting(timePassed,"Robert Cross","CPU",true);
	return result;
}

/**
 * Aplica el algoritmo de Laplacian Of Gaussian en CPU a una imagen.
 */
QImage ImageAlgorythmBusiness::LaplacianOfGaussianFilterCPU(
		QImage imagenEntrada,
		LaplacianOfGaussianImageFilter laplacianOfGaussianFilter) {

	Controlador *controlador = Controlador::Instance();
	//*******TIMER
	QTime *time = new QTime();
	time->start();
	//*******
	//Variables
	//Variables para el ancho y el alto de la imagen original.
	int srcWidth;
	int srcHeight;
	//Imagen auxiliar para procesar el filtro de Gauss
	QImage auxiliar;
	//Imagen resultado de la aplicación del filtro.
	QImage procesada;
	//----------------------------
	//Ancho y alto de la imagen fuente.
	srcWidth = imagenEntrada.width();
	srcHeight = imagenEntrada.height();
	//Instanciación de las imágenes con el ancho el alto y el formato de la imagen original.
	auxiliar = QImage(srcWidth, srcHeight, QImage::Format_RGB32);
	procesada = QImage(srcWidth, srcHeight, QImage::Format_RGB32);
	//Aplicación del filtro de Gauss.
	auxiliar = this->Gauss(imagenEntrada,
			laplacianOfGaussianFilter.getRadioGauss(),
			laplacianOfGaussianFilter.getSigmaGauss());
	//Aplicación del filtro de Laplace.
	procesada = LaplaceFilterCPU(auxiliar,
			laplacianOfGaussianFilter.getLaplaceFilter());
	//Retorno de la imagen resultado de la aplicación del filtro.
	int timePassed = time->elapsed();
	controlador->setCpuExecutionTime(timePassed);
	return procesada;
}

/**
 * Aplica el algoritmo de Sobel Square en CPU a una imagen.
 */
QImage ImageAlgorythmBusiness::SobelSquareFilterCPU(QImage imagenEntrada,
		SobelSquareImageFilter sobelSquareFilter) {
	//*******TIMER
	QTime *time = new QTime();
	time->start();
	//*******
	//Variables
	//Ancho de la imagen de entrada del algoritmo.
	int srcWidth = imagenEntrada.width();
	//Alto de la imagen de entrada del algoritmo.
	int srcHeight = imagenEntrada.height();
	//Imagen resultado del paso a escala de grises.
	QImage grayImg;

	//Radio del filtro
	int filterRadius = sobelSquareFilter.getFilterMatrixX().getWidth() - 1;

	//Se pasa la imagen original a escala de grises con 2 pixels de padding(filtro 3x3).
	grayImg = this->grayScale(imagenEntrada, filterRadius);

	//Se retorna la imagen final.
	QImage result;
	result =  convolutionCPU(srcWidth, srcHeight, grayImg, sobelSquareFilter);
	int timePassed = time->elapsed();
	manageExecutionTimeWriting(timePassed,"Sobel Square","CPU",true);
	return result;
}

/**
 * Aplica el algoritmo de Canny en CPU a una imagen.
 */
QImage ImageAlgorythmBusiness::CannyFilterCPU(QImage imagenEntrada,
		CannyImageFilter cannyFilter) {
	//*******TIMER
	QTime *time = new QTime();
	time->start();
	//*******
	CImage * mPicture;
	QImage image;
	mPicture = new CImage(imagenEntrada);

	//Parte slotCanny::
	mPicture->canny(cannyFilter.getSigmaGauss(), true, true, true);

	//Parte llamada a slotUpdate::
	double low = cannyFilter.getLowerThreshold();
	double high = cannyFilter.getHigherThreshold();

	if (cannyFilter.getHisteresis()) {
		mPicture->useHysteresis(low, high);
	} else {
		mPicture->useSuppressed();
	}
	//llamada a slotRedisplay
	//******
	//Parte slotRedisplay
	int timePassed = time->elapsed();
	manageExecutionTimeWriting(timePassed,"Canny","CPU",true);

	image = *mPicture->mImage;
	mPicture->~CImage();
	return image;
}

//***************************************************************
//		EJECUCIÓN EN GPU: Implementaciones de filtro de detección de imagen en GPU (memoria GLOBAL)
//***************************************************************
/**
 * Aplica el algoritmo de Sobel en GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::SobelFilterGPU(QImage imagenEntrada) {

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc(3 * sizeof(float));
	kernelx2 = (float*) malloc(3 * sizeof(float));
	kernely1 = (float*) malloc(3 * sizeof(float));
	kernely2 = (float*) malloc(3 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	intermedia1 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height());
	intermedia2 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	// 			(1)							(1	 0	-1)
	//		Ax=	(2)  Bx= (1,0,-1)   Ax*Bx=	(2	 0 	-2)= Sobelx
	//			(1)							(1	 0	-1)
	//
	// 			(-1)						(-1	-2	-1)
	//		Ay=	( 0)  By= (1,2,1)   Ay*By=	( 0	 0 	 0)= Sobely
	//			( 1)						( 1	 2	 1)

	kernelx1[0] = 1.0;
	kernelx1[1] = 2.0;
	kernelx1[2] = 1.0;

	kernelx2[0] = 1.0;
	kernelx2[1] = 0.0;
	kernelx2[2] = -1.0;

	kernely1[0] = -1.0;
	kernely1[1] = 0.0;
	kernely1[2] = 1.0;

	kernely2[0] = 1.0;
	kernely2[1] = 2.0;
	kernely2[2] = 1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	CudaAlgorythmBusiness cudaBusiness = CudaAlgorythmBusiness();

	//Llamada para la convolución en X para el filtro de sobel
	intermedia1 = cudaBusiness.GaussCUDA(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx1, kernelx2);
	//Llamada para la convolución en Y para el filtro de sobel
	intermedia2 = cudaBusiness.GaussCUDA(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernely1, kernely2);

	//Interpretación de los resultados mediante el cálculo del gradiente de las dos componentes X e Y.
	for (int i = 0; i < imagenEntrada.width(); i++) {
		for (int j = 0; j < imagenEntrada.height(); j++) {
			matrizSalida[i + j * imagenEntrada.width()] = sqrt(
					(intermedia1[i + j * imagenEntrada.width()] * intermedia1[i
							+ j * imagenEntrada.width()]) + (intermedia2[i + j
							* imagenEntrada.width()] * intermedia2[i + j
							* imagenEntrada.width()]));

		}
	}
	//Devolución de la imagen como resultado.
	QImage result;
	 result = this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
	 int  timePassed =time->elapsed();
	 manageExecutionTimeWriting(timePassed,"Sobel","GPU",false);
	 return result;
}

/**
 * Aplica el algoritmo de Prewitt en GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::PrewittFilterGPU(QImage imagenEntrada) {

	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc(3 * sizeof(float));
	kernelx2 = (float*) malloc(3 * sizeof(float));
	kernely1 = (float*) malloc(3 * sizeof(float));
	kernely2 = (float*) malloc(3 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	intermedia1 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height());
	intermedia2 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	//
	// 			(1)							( 1	 0  -1)
	//		Ax=	(1)  Bx= (1,0,-1)   Ax*Bx=	( 1	 0  -1)= Prewittx
	//			(1)							( 1	 0  -1)
	//
	// 			(-1)						(-1 -1	-1)
	//		Ay=	( 0)  By= (1,1,1)   Ay*By=	( 0	 0 	 0)= Prewitty
	//			( 1)						( 1	 1	 1)

	kernelx1[0] = 1.0;
	kernelx1[1] = 1.0;
	kernelx1[2] = 1.0;

	kernelx2[0] = 1.0;
	kernelx2[1] = 0.0;
	kernelx2[2] = -1.0;

	kernely1[0] = -1.0;
	kernely1[1] = 0.0;
	kernely1[2] = 1.0;

	kernely2[0] = 1.0;
	kernely2[1] = 1.0;
	kernely2[2] = 1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	CudaAlgorythmBusiness cudaBusiness = CudaAlgorythmBusiness();

	//Llamada para la convolución en X para el filtro de sobel
	intermedia1 = cudaBusiness.GaussCUDA(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx1, kernelx2);
	//Llamada para la convolución en Y para el filtro de sobel
	intermedia2 = cudaBusiness.GaussCUDA(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernely1, kernely2);

	//Interpretación de los resultados mediante el cálculo del gradiente de las dos componentes X e Y.
	for (int i = 0; i < imagenEntrada.width(); i++) {
		for (int j = 0; j < imagenEntrada.height(); j++) {
			matrizSalida[i + j * imagenEntrada.width()] = sqrt(
					(intermedia1[i + j * imagenEntrada.width()] * intermedia1[i
							+ j * imagenEntrada.width()]) + (intermedia2[i + j
							* imagenEntrada.width()] * intermedia2[i + j
							* imagenEntrada.width()]));

		}
	}
	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
}

/**
 * Aplica el algoritmo de Laplace en GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::LaplaceFilterGPU(QImage imagenEntrada) {

	float *matrizEntrada;
	float *matrizSalida;
	float *kernelx;
	float *kernely;
	//Reservas de memorias para los kernels
	kernelx = (float*) malloc(25 * sizeof(float));
	kernely = (float*) malloc(25 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	/* Filtro de Laplace */
	kernelx[0] = -1.0;
	kernelx[1] = -1.0;
	kernelx[2] = -1.0;
	kernelx[3] = -1.0;
	kernelx[4] = -1.0;
	kernelx[5] = -1.0;
	kernelx[6] = -1.0;
	kernelx[7] = -1.0;
	kernelx[8] = -1.0;
	kernelx[9] = -1.0;
	kernelx[10] = -1.0;
	kernelx[11] = -1.0;
	kernelx[12] = 24;
	kernelx[13] = -1.0;
	kernelx[14] = -1.0;
	kernelx[15] = -1.0;
	kernelx[16] = -1.0;
	kernelx[17] = -1.0;
	kernelx[18] = -1.0;
	kernelx[19] = -1.0;
	kernelx[20] = -1.0;
	kernelx[21] = -1.0;
	kernelx[22] = -1.0;
	kernelx[23] = -1.0;
	kernelx[24] = -1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	Cuda2dConvolutionBusiness cudaBusiness = Cuda2dConvolutionBusiness();

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

	//Llamada para la convolución en X para el filtro de sobel
	matrizSalida = cudaBusiness.ConvolveLaplace(matrizEntrada,
			imagenEntrada.width(), imagenEntrada.height(), kernelx, 25);

	int timed = time->elapsed();
	Controlador *controlador = Controlador::Instance();
	controlador->setGpuExecutionTime(timed);

	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());

}

/**
 * Aplica el algoritmo de Robert Cross en GPU (memoria global) sobre una imagen.
 */

QImage ImageAlgorythmBusiness::RobertCrossFilterGPU(QImage imagenEntrada) {

	float *matrizEntrada;
	float *matrizSalida;
	float *kernelx;
	float *kernely;
	//Reservas de memorias para los kernels
	kernelx = (float*) malloc(9 * sizeof(float));
	kernely = (float*) malloc(9 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	kernelx[0] = 0.0;
	kernelx[1] = 0.0;
	kernelx[2] = -1.0;
	kernelx[3] = 0.0;
	kernelx[4] = 1.0;
	kernelx[5] = 0.0;
	kernelx[6] = 0.0;
	kernelx[7] = 0.0;
	kernelx[8] = 0.0;

	kernely[0] = -1.0;
	kernely[1] = 0.0;
	kernely[2] = 0.0;
	kernely[3] = 0.0;
	kernely[4] = 1.0;
	kernely[5] = 0.0;
	kernely[6] = 0.0;
	kernely[7] = 0.0;
	kernely[8] = 0.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	Cuda2dConvolutionBusiness cudaBusiness = Cuda2dConvolutionBusiness();

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	//Llamada para la convolución en X e Y para filtro de robert cross
	matrizSalida = cudaBusiness.Convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx, kernely, 9);
	int timed = time->elapsed();
	Controlador *controlador = Controlador::Instance();
	controlador->setGpuExecutionTime(timed);

	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
}

/**
 * Aplica el algoritmo de Laplacian Of Gaussian GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::LaplacianOfGaussianFilterGPU(
		QImage imagenEntrada,LaplacianOfGaussianImageFilter laplacianOfGaussianFilter) {

	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	float *kernelx;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc((laplacianOfGaussianFilter.getRadioGauss()*2+1) * sizeof(float));
	kernelx2 = (float*) malloc((laplacianOfGaussianFilter.getRadioGauss()*2+1) * sizeof(float));
	kernely1 = (float*) malloc((laplacianOfGaussianFilter.getRadioGauss()*2+1) * sizeof(float));
	kernely2 = (float*) malloc((laplacianOfGaussianFilter.getRadioGauss()*2+1) * sizeof(float));
	kernelx = (float*) malloc(25 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	intermedia1 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height());
	intermedia2 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));


	kernelx1=this->generateGaussianKernels(laplacianOfGaussianFilter.getSigmaGauss(),laplacianOfGaussianFilter.getRadioGauss());


	/* Filtro de Laplace */
	kernelx[0] = -1.0;
	kernelx[1] = -1.0;
	kernelx[2] = -1.0;
	kernelx[3] = -1.0;
	kernelx[4] = -1.0;
	kernelx[5] = -1.0;
	kernelx[6] = -1.0;
	kernelx[7] = -1.0;
	kernelx[8] = -1.0;
	kernelx[9] = -1.0;
	kernelx[10] = -1.0;
	kernelx[11] = -1.0;
	kernelx[12] = 24;
	kernelx[13] = -1.0;
	kernelx[14] = -1.0;
	kernelx[15] = -1.0;
	kernelx[16] = -1.0;
	kernelx[17] = -1.0;
	kernelx[18] = -1.0;
	kernelx[19] = -1.0;
	kernelx[20] = -1.0;
	kernelx[21] = -1.0;
	kernelx[22] = -1.0;
	kernelx[23] = -1.0;
	kernelx[24] = -1.0;

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

		matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
		CudaConvolutionSeparableBusiness cudaBusiness = CudaConvolutionSeparableBusiness();

		//Llamada para la convolución en X e Y para filtro de robert cross
		matrizSalida = cudaBusiness.Convolve(matrizEntrada, imagenEntrada.width(),
				imagenEntrada.height(), kernelx1, kernelx1, (laplacianOfGaussianFilter.getRadioGauss()*2+1));

		Cuda2dConvolutionBusiness cudaBusinessLaplace = Cuda2dConvolutionBusiness();

		//Llamada para la convolución en X para el filtro de sobel
			matrizSalida = cudaBusinessLaplace.ConvolveLaplace(matrizSalida,
					imagenEntrada.width(), imagenEntrada.height(), kernelx, 25);
		int timed = time->elapsed();
		Controlador *controlador = Controlador::Instance();
		controlador->setGpuExecutionTime(timed);
		//Devolución de la imagen como resultado.
		return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
				imagenEntrada.height());

}

/**
 * Aplica el algoritmo de Sobel Square en GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::SobelSquareFilterGPU(QImage imagenEntrada) {

	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc(3 * sizeof(float));
	kernelx2 = (float*) malloc(3 * sizeof(float));
	kernely1 = (float*) malloc(3 * sizeof(float));
	kernely2 = (float*) malloc(3 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	intermedia1 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height());
	intermedia2 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	// 			(1)							(1	 0	-1)
	//		Ax=	(2)  Bx= (1,0,-1)   Ax*Bx=	(2	 0 	-2)= Sobelx
	//			(1)							(1	 0	-1)
	//
	// 			(-1)						(-1	-2	-1)
	//		Ay=	( 0)  By= (1,2,1)   Ay*By=	( 0	 0 	 0)= Sobely
	//			( 1)						( 1	 2	 1)

	kernelx1[0] = 1.0;
	kernelx1[1] = 2.0;
	kernelx1[2] = 1.0;

	kernelx2[0] = 1.0;
	kernelx2[1] = 0.0;
	kernelx2[2] = -1.0;

	kernely1[0] = -1.0;
	kernely1[1] = 0.0;
	kernely1[2] = 1.0;

	kernely2[0] = 1.0;
	kernely2[1] = 2.0;
	kernely2[2] = 1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	CudaAlgorythmBusiness cudaBusiness = CudaAlgorythmBusiness();

	//Llamada para la convolución en X para el filtro de sobel
	intermedia1 = cudaBusiness.GaussCUDA(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx1, kernelx2);
	//Llamada para la convolución en Y para el filtro de sobel
	intermedia2 = cudaBusiness.GaussCUDA(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernely1, kernely2);

	//Interpretación de los resultados mediante el cálculo del gradiente de las dos componentes X e Y.
	for (int i = 0; i < imagenEntrada.width(); i++) {
		for (int j = 0; j < imagenEntrada.height(); j++) {
			matrizSalida[i + j * imagenEntrada.width()] = ((intermedia1[i + j
					* imagenEntrada.width()] * intermedia1[i + j
					* imagenEntrada.width()]) + (intermedia2[i + j
					* imagenEntrada.width()] * intermedia2[i + j
					* imagenEntrada.width()])) / 8;

		}
	}
	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());

}

/**
 * Gestiona la escritura de valores de tiempo de ejecución según el modo de la aplicación
 * y el modo de ejecución de la aplicación y el algoritmo seleccionado.
 *
 */
void ImageAlgorythmBusiness::manageExecutionTimeWriting(int executionTime,QString processName,QString mode,bool cpu) {
	Constants *constants = Constants::Instance();
	Controlador *controlador = Controlador::Instance();
	TimeContainer container;

	container.setProcess(processName);
	container.setExecutionType(mode);

	//Se informan en el controlador los datos de :
		//1-Algoritmo
		//2-Modo de ejecución (CPU o GPU).
		//3-Tiempo de ejecución.
	//Se sale. Se delegará en la llamada al controlador la composición del TimeContainer y su inserción en la lista correcta.

	controlador->setTimeAlgorythm(processName);
	if(cpu){

		if(controlador->getCpuExecutionTime()!=NULL && controlador->getCpuExecutionTime()!=0){
			executionTime = executionTime + controlador->getCpuExecutionTime();
		}
		controlador->setCpuExecutionTime(executionTime);
	}else{
		controlador->setGpuExecutionTime(executionTime);
	}


}

//***************************************************************
//		EJECUCIÓN EN GPU: Implementaciones de filtro de detección de imagen en GPU (MEMORIA COMPARTIDA)
//***************************************************************
//**********************
//**********************

/**
 * Aplica el algoritmo de Sobel en GPU (memoria COMPARTIDA) sobre una imagen
 */
QImage ImageAlgorythmBusiness::SobelFilterGPUShared(QImage imagenEntrada) {

	Controlador *controlador = Controlador::Instance();
	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc(3 * sizeof(float));
	kernelx2 = (float*) malloc(3 * sizeof(float));
	kernely1 = (float*) malloc(3 * sizeof(float));
	kernely2 = (float*) malloc(3 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	// 			(1)							(1	 0	-1)
	//		Ax=	(2)  Bx= (1,0,-1)   Ax*Bx=	(2	 0 	-2)= Sobelx
	//			(1)							(1	 0	-1)
	//
	// 			(-1)						(-1	-2	-1)
	//		Ay=	( 0)  By= (1,2,1)   Ay*By=	( 0	 0 	 0)= Sobely
	//			( 1)						( 1	 2	 1)

	kernelx1[0] = 1.0;
	kernelx1[1] = 2.0;
	kernelx1[2] = 1.0;

	kernelx2[0] = 1.0;
	kernelx2[1] = 0.0;
	kernelx2[2] = -1.0;

	kernely1[0] = -1.0;
	kernely1[1] = 0.0;
	kernely1[2] = 1.0;

	kernely2[0] = 1.0;
	kernely2[1] = 2.0;
	kernely2[2] = 1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	Cuda5StepConvolutionBusiness cudaBusiness = Cuda5StepConvolutionBusiness();
	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	//Llamada para la convolución en X para el filtro de sobel
	intermedia1 = cudaBusiness.convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx1, kernelx2,1);
	//Llamada para la convolución en Y para el filtro de sobel
	intermedia2 = cudaBusiness.convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernely1, kernely2,1);
	 int  timePassed =time->elapsed();

	//Interpretación de los resultados mediante el cálculo del gradiente de las dos componentes X e Y.
	for (int i = 0; i < imagenEntrada.width(); i++) {
		for (int j = 0; j < imagenEntrada.height(); j++) {
			matrizSalida[i + j * imagenEntrada.width()] = sqrt(
					(intermedia1[i + j * imagenEntrada.width()] * intermedia1[i
							+ j * imagenEntrada.width()]) + (intermedia2[i + j
							* imagenEntrada.width()] * intermedia2[i + j
							* imagenEntrada.width()]));

		}
	}
	//Devolución de la imagen como resultado.
	QImage result;
	 result = this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
	 //Liberación de memoria de las matrices auxiliares generadas.
	 free(intermedia1);
	 free(intermedia2);
	 free(matrizSalida);
	 free(matrizEntrada);
	 //Libreación de memoria de los kernels generados.
	 free(kernelx1);
	 free(kernelx2);
	 free(kernely1);
	 free(kernely2);

	 return result;
}

/**
 * Aplica el algoritmo de Sobel Square en GPU (memoria COMPARTIDA) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::SobelSquareFilterGPUShared(QImage imagenEntrada){
	Controlador *controlador = Controlador::Instance();

	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc(3 * sizeof(float));
	kernelx2 = (float*) malloc(3 * sizeof(float));
	kernely1 = (float*) malloc(3 * sizeof(float));
	kernely2 = (float*) malloc(3 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	intermedia1 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height());
	intermedia2 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	matrizEntrada = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	// 			(1)							(1	 0	-1)
	//		Ax=	(2)  Bx= (1,0,-1)   Ax*Bx=	(2	 0 	-2)= Sobelx
	//			(1)							(1	 0	-1)
	//
	// 			(-1)						(-1	-2	-1)
	//		Ay=	( 0)  By= (1,2,1)   Ay*By=	( 0	 0 	 0)= Sobely
	//			( 1)						( 1	 2	 1)

	kernelx1[0] = 1.0;
	kernelx1[1] = 2.0;
	kernelx1[2] = 1.0;

	kernelx2[0] = 1.0;
	kernelx2[1] = 0.0;
	kernelx2[2] = -1.0;

	kernely1[0] = -1.0;
	kernely1[1] = 0.0;
	kernely1[2] = 1.0;

	kernely2[0] = 1.0;
	kernely2[1] = 2.0;
	kernely2[2] = 1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada,matrizEntrada);
	Cuda5StepConvolutionBusiness cudaBusiness = Cuda5StepConvolutionBusiness();

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

	//Llamada para la convolución en X para el filtro de sobel
	intermedia1 = cudaBusiness.convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx1, kernelx2,1);
	//Llamada para la convolución en Y para el filtro de sobel
	intermedia2 = cudaBusiness.convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernely1, kernely2,1);

	 int  timePassed =time->elapsed();
	 //controlador->setGpuExecutionTime(timePassed);

	//Interpretación de los resultados mediante el cálculo del gradiente de las dos componentes X e Y.
	for (int i = 0; i < imagenEntrada.width(); i++) {
		for (int j = 0; j < imagenEntrada.height(); j++) {
			matrizSalida[i + j * imagenEntrada.width()] = ((intermedia1[i + j
					* imagenEntrada.width()] * intermedia1[i + j
					* imagenEntrada.width()]) + (intermedia2[i + j
					* imagenEntrada.width()] * intermedia2[i + j
					* imagenEntrada.width()])) / 8;

		}
	}
	//Devolución de la imagen como resultado.
	QImage result;
	 result = this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
	 return result;
}

/**
 * Aplica el algoritmo de Prewitt en GPU (memoria COMPARTIDA) sobr euna imagen.
 */
QImage ImageAlgorythmBusiness::PrewittFilterGPUShared(QImage imagenEntrada) {

	Controlador *controlador = Controlador::Instance();

	float *matrizEntrada;
	float *matrizSalida;
	float *intermedia1;
	float *intermedia2;
	float *kernelx1;
	float *kernelx2;
	float *kernely1;
	float *kernely2;
	//Reservas de memorias para los kernels
	kernelx1 = (float*) malloc(3 * sizeof(float));
	kernelx2 = (float*) malloc(3 * sizeof(float));
	kernely1 = (float*) malloc(3 * sizeof(float));
	kernely2 = (float*) malloc(3 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));
	intermedia1 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height());
	intermedia2 = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	// 			(1)							(1	 0	-1)
	//		Ax=	(2)  Bx= (1,0,-1)   Ax*Bx=	(2	 0 	-2)= Sobelx
	//			(1)							(1	 0	-1)
	//
	// 			(-1)						(-1	-2	-1)
	//		Ay=	( 0)  By= (1,2,1)   Ay*By=	( 0	 0 	 0)= Sobely
	//			( 1)						( 1	 2	 1)

	kernelx1[0] = 1.0;
	kernelx1[1] = 1.0;
	kernelx1[2] = 1.0;

	kernelx2[0] = 1.0;
	kernelx2[1] = 0.0;
	kernelx2[2] = -1.0;

	kernely1[0] = -1.0;
	kernely1[1] = 0.0;
	kernely1[2] = 1.0;

	kernely2[0] = 1.0;
	kernely2[1] = 1.0;
	kernely2[2] = 1.0;

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	Cuda5StepConvolutionBusiness cudaBusiness = Cuda5StepConvolutionBusiness();

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

	//Llamada para la convolución en X para el filtro de sobel
	intermedia1 = cudaBusiness.convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernelx1, kernelx2,1);
	//Llamada para la convolución en Y para el filtro de sobel
	intermedia2 = cudaBusiness.convolve(matrizEntrada, imagenEntrada.width(),
			imagenEntrada.height(), kernely1, kernely2,1);

	 int  timePassed =time->elapsed();
	 //controlador->setGpuExecutionTime(timePassed);

	//Interpretación de los resultados mediante el cálculo del gradiente de las dos componentes X e Y.
	for (int i = 0; i < imagenEntrada.width(); i++) {
		for (int j = 0; j < imagenEntrada.height(); j++) {
			matrizSalida[i + j * imagenEntrada.width()] = sqrt(
					(intermedia1[i + j * imagenEntrada.width()] * intermedia1[i
							+ j * imagenEntrada.width()]) + (intermedia2[i + j
							* imagenEntrada.width()] * intermedia2[i + j
							* imagenEntrada.width()]));

		}
	}
	//Devolución de la imagen como resultado.
	QImage result;
	 result = this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
	 return result;
}

QImage ImageAlgorythmBusiness::cannyGpuShared(QImage imagenEntrada,CannyImageFilter cannyFilter) {

	Controlador *controlador = Controlador::Instance();
	CudaInvoquer invoquer = CudaInvoquer();

	float *matrizEntrada;
	float *matrizSalida;
	QImage resultImage;

	//Traducción de la imagen de entrada a puntero a float.
	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);

	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********

	matrizSalida = invoquer.invoque(matrizEntrada,imagenEntrada.width(),imagenEntrada.height(),cannyFilter.getRadioGauss(),cannyFilter.getSigmaGauss(),cannyFilter.getLowerThreshold()+5,cannyFilter.getHigherThreshold()+7,cannyFilter.getHisteresis());
	int  timePassed =time->elapsed();

	//Conversión de la matriz de salida a QImage.
	resultImage = this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
					imagenEntrada.height());

	return resultImage;


}


QImage ImageAlgorythmBusiness::RobertCrossFilterGPUShared(QImage imagenEntrada) {
	Controlador *controlador = Controlador::Instance();
	float *matrizEntrada;
	float *matrizSalida;
	float *kernelx;
	//Reservas de memorias para los kernels
	kernelx = (float*) malloc(25 * sizeof(float));

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	CudaRobertCrossShared cudaBusiness = CudaRobertCrossShared();
	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	matrizSalida = cudaBusiness.convolve(matrizEntrada,imagenEntrada.width(), imagenEntrada.height(), 1);
	int  timePassed =time->elapsed();
	controlador->setGpuExecutionTime(timePassed);
	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());
}

/**
 * Aplica el algoritmo de Laplace en GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::LaplaceFilterGPUShared(QImage imagenEntrada) {

	Controlador *controlador = Controlador::Instance();
	float *matrizEntrada;
	float *matrizSalida;

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	CudaLaplaceShared cudaBusiness = CudaLaplaceShared();
	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	matrizSalida = cudaBusiness.convolve(matrizEntrada,imagenEntrada.width(), imagenEntrada.height(), 1);
	int  timePassed =time->elapsed();
	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());

}

/**
 * Aplica el algoritmo de Laplace en GPU (memoria global) sobre una imagen.
 */
QImage ImageAlgorythmBusiness::LaplacianOfGaussianFilterGPUShared(QImage imagenEntrada,LaplacianOfGaussianImageFilter laplacianOfGaussianFilter) {

	Controlador *controlador = Controlador::Instance();
	float *matrizEntrada;
	float *matrizSalida;

	matrizSalida = (float*) malloc(
			imagenEntrada.width() * imagenEntrada.height() * sizeof(float));

	matrizEntrada = this->fromQImageToMatrix(imagenEntrada);
	CudaLaplaceShared cudaBusiness = CudaLaplaceShared();
	//******TIMER
	QTime *time = new QTime();
	time->start();
	//********
	matrizSalida = cudaBusiness.laplacianOfGaussianShared(matrizEntrada,imagenEntrada.width(), imagenEntrada.height(), 1,laplacianOfGaussianFilter.getRadioGauss(),laplacianOfGaussianFilter.getSigmaGauss());
	int  timePassed =time->elapsed();
	//Devolución de la imagen como resultado.
	return this->fromMatrixToQImage(matrizSalida, imagenEntrada.width(),
			imagenEntrada.height());

}
