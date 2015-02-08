/*
 * CudaInvoquer.cuh
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

#ifndef CUDAINVOQUER_H_
#define CUDAINVOQUER_H_

class CudaInvoquer {
public:
	CudaInvoquer();
	virtual ~CudaInvoquer();
	float* invoque(float* imagenEntrada,int imagen_width,int imagen_height,int radius, float sigma,float lowerThreshold, float upperThreshold,bool histeresis);
};

#endif /* CUDAINVOQUER_H_ */
