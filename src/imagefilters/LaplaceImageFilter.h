/*
 * LaplaceImageFilter.h
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Laplace.
 */

#ifndef LAPLACEIMAGEFILTER_H_
#define LAPLACEIMAGEFILTER_H_

#include "ImageFilter.h"

class LaplaceImageFilter : public ImageFilter {
public:
	LaplaceImageFilter();
	virtual ~LaplaceImageFilter();
};

#endif /* LAPLACEIMAGEFILTER_H_ */
