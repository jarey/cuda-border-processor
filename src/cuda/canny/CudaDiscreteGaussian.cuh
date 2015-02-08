/*
 * CudaDiscreteGaussian.cuh
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase encargada de la implementación en GPU memorial global de la creación del filtro gaussiano con los parámetros daos y del difuminado
 * de la imagen a partir de dicho filtro.
 */

#ifndef CUDADISCRETEGAUSSIAN_H_
#define CUDADISCRETEGAUSSIAN_H_

class CudaDiscreteGaussian {
public:
	CudaDiscreteGaussian();
	virtual ~CudaDiscreteGaussian();
	float* cuda1DGaussianOperator(dim3 DimGrid, dim3 DimBlock, unsigned int width, float gaussianVariance);
};

#endif /* CUDADISCRETEGAUSSIAN_CUH_ */



