/*
 * SobelSquareImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Sobel Square.
 */

#ifndef SOBELSQUAREIMAGEFILTER_H_
#define SOBELSQUAREIMAGEFILTER_H_

#include "ImageFilter.h"

class SobelSquareImageFilter : public ImageFilter{
public:
	SobelSquareImageFilter();
	virtual ~SobelSquareImageFilter();
};

#endif /* SOBELSQUAREIMAGEFILTER_H_ */
