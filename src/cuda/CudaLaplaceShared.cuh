/*
 * CudaLaplaceShared.cuh
 *
 *  Created on: 07/01/2013
 *      Author: jose
 */

#ifndef CUDALAPLACESHARED_H_
#define CUDALAPLACESHARED_H_


typedef unsigned char Pixel;

// enumerado para selección de modalidad de ejecución
enum SobelDisplayMode {
	SOBELDISPLAY_IMAGE = 0,
	SOBELDISPLAY_SOBELTEX,
	SOBELDISPLAY_SOBELSHARED
};

static char *filterMode[] = {
	"Original Filter",
	"Sobel Shared",
	"Sobel Shared+Texture",
	NULL
};

extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, float fScale);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);

class CudaLaplaceShared {
public:
	CudaLaplaceShared();
	virtual ~CudaLaplaceShared();
	float* convolve(float *imagen, int ancho,int alto, int brillo);
	float* laplacianOfGaussianShared(float *imagen, int ancho,int alto, int brillo,int radius, float sigma);
};

#endif
