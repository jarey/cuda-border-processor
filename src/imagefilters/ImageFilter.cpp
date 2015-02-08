/*
 * ImageFilter.cpp
 *
 * Creado: 23/03/2012
 * Author: jose
 * Descripción: clase padre de los filtros de imagen. Contiene los atributos y operaciones
 * comunes a todos los filtros de imágenes.
 */

#include "ImageFilter.h"
#include "./src/common/FloatMatrix.h"

ImageFilter::ImageFilter() {
	// TODO Auto-generated constructor stub

}

ImageFilter::ImageFilter(QString filterName, FloatMatrix filterMatrixX,
		FloatMatrix filterMatrixY) {

	this->setFilterName(filterName);
	this->setFilterMatrixX(filterMatrixX);
	this->setFilterMatrixY(filterMatrixY);
}

ImageFilter::~ImageFilter() {
	// TODO Auto-generated destructor stub
}

//Implementaciones para getters
//Getters
QString ImageFilter::getFilterName() {
	return this->filterName;
}
FloatMatrix ImageFilter::getFilterMatrixX() {
	return this->filterMatrixX;
}
FloatMatrix ImageFilter::getFilterMatrixY() {
	return this->filterMatrixY;
}

//Implementaciones para setters
//Setters
void ImageFilter::setFilterName(QString filterName) {
	this->filterName = filterName;
}
void ImageFilter::setFilterMatrixX(FloatMatrix filterX) {
	this->filterMatrixX = filterX;
}
void ImageFilter::setFilterMatrixY(FloatMatrix filterY) {
	this->filterMatrixY = filterY;
}

