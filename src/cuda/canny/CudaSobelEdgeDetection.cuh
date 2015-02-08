/*
 * CudaSobelEdgeDetection.cuh
 *
 * Creado: 11/11/2012
 * Autor: jose
 * Descripción: clase que implementa el filtrado a través de la ventana de sobel de forma separable y en memoria compartida en la GPU.
 */

#ifndef CUDASOBELEDGEDETECTION_H_
#define CUDASOBELEDGEDETECTION_H_

template <class TMagnitude, class TDirection>class Gradient
{
public:
  TMagnitude *Magnitude;
  TDirection *Direction;
};

typedef Gradient<float,float>  Tgrad;

class CudaSobelEdgeDetection {
public:
	CudaSobelEdgeDetection();
	virtual ~CudaSobelEdgeDetection();
	Tgrad* cudaSobel(Tgrad* d_gradient, const float *d_img, int width, int height);
	float* cudaSobelM(Tgrad *d_gradient,float *d_img, int width, int height);
};

#endif /* CUDASOBELEDGEDETECTION_CUH_ */
