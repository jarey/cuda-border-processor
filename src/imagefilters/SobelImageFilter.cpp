/*
 * SobelImageFilter.cpp
 *
 * Creado: 23/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Sobel.
 */

#include "SobelImageFilter.h"
#include "./src/common/Constants.h"

SobelImageFilter::SobelImageFilter() {
	Constants *constants = Constants::Instance();

	this->setFilterName(constants->getSobelConstant());
	this->setFilterMatrixX(constants->getSobelX());
	this->setFilterMatrixY(constants->getSobelY());
}


SobelImageFilter::~SobelImageFilter() {
//	delete ImageFilter::filtername;
//	delete this->getFilterMatrixX();
//	delete this->getFilterMatrixY();
}
