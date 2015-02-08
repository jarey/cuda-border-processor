/*
 * Cuda2DSeparableConvolution.cuh
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase encargada de realizar la convolución separable en memoria compartida de una imagen.
 */

#ifndef CUDA2DSEPARABLECONVOLUTION_H_
#define CUDA2DSEPARABLECONVOLUTION_H_

class Cuda2DSeparableConvolution {
public:
	Cuda2DSeparableConvolution();
	virtual ~Cuda2DSeparableConvolution();
	float* cuda2DSeparableConvolution(dim3 DimGrid, dim3 DimBlock, const float *d_img, int width, int height, const float *d_kernelH, int sizeH, const float *d_kernelV, int sizeV);
};

#endif /* CUDA2DSEPARABLECONVOLUTION_H_ */
