/*
 * CudaLaplaceShared.cuh
 *
 *  Created on: 07/01/2013
 *      Author: jose
 */

#ifndef CUDAROBERTCROSSSHARED_H_
#define CUDAROBERTCROSSSHARED_H_


typedef unsigned char Pixel;

extern "C" void robertCrossFilter(Pixel *odata, int iw, int ih, float fScale);
extern "C" void setupTextureRobert(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTextureRobert(void);
extern "C" void initFilterRobert(void);

class CudaRobertCrossShared {
public:
	CudaRobertCrossShared();
	virtual ~CudaRobertCrossShared();
	float* convolve(float *imagen, int ancho,int alto, int brillo);
};

#endif

