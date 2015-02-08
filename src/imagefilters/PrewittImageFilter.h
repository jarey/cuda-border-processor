/*
 * PrewittImageFilter.h
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Prewitt.
 */

#ifndef PREWITTIMAGEFILTER_H_
#define PREWITTIMAGEFILTER_H_

#include "ImageFilter.h"

class PrewittImageFilter : public ImageFilter{
public:
	PrewittImageFilter();
	virtual ~PrewittImageFilter();
};

#endif /* PREWITTIMAGEFILTER_H_ */
