/*
 * CudaZeroCrossing.cuh
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: Clase que implementa en GPU el cruce a 0 de una imagen. Paso incluido en la implementación dada para el proceso de Canny
 * en GPU memoria compartida en lugar del proceso de supresión de no-maximos.
 */

#ifndef CUDAZEROCROSSING_H_
#define CUDAZEROCROSSING_H_

class CudaZeroCrossing {
public:
	CudaZeroCrossing();
	virtual ~CudaZeroCrossing();
	float* cudaZeroCrossing(dim3 DimGrid, dim3 DimBlock, float *d_input, int width, int height);
};

#endif /* CUDAZEROCROSSING_H_ */
