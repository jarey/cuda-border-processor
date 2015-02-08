/*
 * CudaCannyEdgeDetection.cuh
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase encargada de realizar el proceso de canny, el proceso de histéresis.
 */

#ifndef CUDACANNYEDGEDETECTION_H_
#define CUDACANNYEDGEDETECTION_H_


// Tipos de valores de píxeles usados para la histéresis.
#define DEFINITIVE_EDGE 255
#define POSSIBLE_EDGE 128
#define NON_EDGE 0

// Tamaños de tiras para la histéresis.
#define SLICE_WIDTH 18
#define SLICE_BLOCK_WIDTH 16


// Constantes para declarar si el pixel ha sido modfiicado o no.
#define MODIFIED 1
#define NOT_MODIFIED 0

class CudaCannyEdgeDetection {
public:
	CudaCannyEdgeDetection();
	virtual ~CudaCannyEdgeDetection();
	float* cuda2ndDerivativePos(dim3 DimGrid, dim3 DimBlock, const float *d_input, const float *d_Lvv, int width, int height);
	float* cuda2ndDerivative(dim3 DimGrid, dim3 DimBlock, const float *d_input, int width, int height);
	float* cudaHysteresis(dim3 DimGrid, dim3 DimBlock, float *d_img, float *d_gauss, int width, int height, float t1, float t2);
};

#endif /* CUDACANNYEDGEDETECTION_H_ */
