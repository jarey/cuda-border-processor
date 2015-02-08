/*
 * SobelImageFilter.h
 *
 * Creado: 23/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Sobel.
 */

#ifndef SOBELIMAGEFILTER_H_
#define SOBELIMAGEFILTER_H_

#include "ImageFilter.h"

class SobelImageFilter : public ImageFilter {
public:
	SobelImageFilter();
	virtual ~SobelImageFilter();
};

#endif /* SOBELIMAGEFILTER_H_ */
