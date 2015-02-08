/*
 * ImageFilter.h
 *
 * Creado: 23/03/2012
 * Author: jose
 * Descripción: clase padre de los filtros de imagen. Contiene los atributos y operaciones
 * comunes a todos los filtros de imágenes.
 */

#ifndef IMAGEFILTER_H_
#define IMAGEFILTER_H_

#include <QString>
#include "./src/common/FloatMatrix.h"

class ImageFilter {
public:
	ImageFilter();
	ImageFilter(QString filterName, FloatMatrix filterMatrixX,
			FloatMatrix filterMatrixY);
	virtual ~ImageFilter();
	//Getters
	QString getFilterName();
	FloatMatrix getFilterMatrixX();
	FloatMatrix getFilterMatrixY();
	//Setters
	void setFilterName(QString filterName);
	void setFilterMatrixX(FloatMatrix filterX);
	void setFilterMatrixY(FloatMatrix filterY);
	//Atributos obligatorios para un filtro de imagen
private:
	//1- Nombre del filtro
	QString filterName;
	//2-Matrix del filtro en cuestión.
	FloatMatrix filterMatrixX;
	FloatMatrix filterMatrixY;
};

#endif /* IMAGEFILTER_H_ */
