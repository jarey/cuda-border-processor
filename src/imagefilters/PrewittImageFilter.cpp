/*
 * PrewittImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Prewitt.
 */

#include "PrewittImageFilter.h"
#include "./src/common/Constants.h"

PrewittImageFilter::PrewittImageFilter() {
	Constants *constants = Constants::Instance();

	this->setFilterName(constants->getPrewittConstant());
	this->setFilterMatrixX(constants->getPrewittX());
	this->setFilterMatrixY(constants->getPrewittY());
}

PrewittImageFilter::~PrewittImageFilter() {
	// TODO Auto-generated destructor stub
}



