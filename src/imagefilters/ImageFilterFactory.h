/*
 * ImageFilterFactory.h
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: factoría de filtros, se emplean apra obtener de forma automática, el filtro asociado
 * a un proceso escogido por pantalla.
 */

#ifndef IMAGEFILTERFACTORY_H_
#define IMAGEFILTERFACTORY_H_

#include "ImageFilter.h"
#include <QString>

class ImageFilterFactory {

public:
	ImageFilterFactory();
	virtual ~ImageFilterFactory();
	ImageFilter *getImageFilterInstance(QString filterName,bool histeresis, float radioGauss,
				double sigmaGauss, double lowerThreshold, double higherThreshold);
};

#endif /* IMAGEFILTERFACTORY_H_ */
