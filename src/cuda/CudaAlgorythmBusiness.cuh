/*
 * CudaAlgorythmBusiness.h
 *
 *  Created on: 29/03/2012
 *      Author: jose
 */

#ifndef CUDAALGORYTHMBUSINESS_H_
#define CUDAALGORYTHMBUSINESS_H_

#define N	10

class CudaAlgorythmBusiness {
public:
	CudaAlgorythmBusiness();
	virtual ~CudaAlgorythmBusiness();
	float* GaussCUDA(float *imagen, int ancho, int alto,float *kernel1,float *kernel2);
};

#endif /* CUDAALGORYTHMBUSINESS_H_ */
