/*
 * ImageProcessingBusiness.h
 *
 * Creado: 24/03/2012
 * Autor: jose
 * Descripciñon: clase encargada de orquestrar la ejecución de los algoritmos de forma automática.
 * Se hace uso del filtro concreto instanciado por pantalla y se asigna el evaluador de filtro
 * concreto, según el filtro haciendo uso de la factoría de evaluadores.
 */

#ifndef IMAGEPROCESSINGBUSINESS_H_
#define IMAGEPROCESSINGBUSINESS_H_

#include "./src/imagefilters/ImageFilter.h"

class ImageProcessingBusiness {
public:
	ImageProcessingBusiness();
	virtual ~ImageProcessingBusiness();
	void doProcess(ImageFilter &filter);
	void doProcess(QList<ImageFilter*> *filters);

};

#endif /* IMAGEPROCESSINGBUSINESS_H_ */
