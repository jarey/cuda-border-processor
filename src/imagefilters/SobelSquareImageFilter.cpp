/*
 * SobelSquareImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Sobel Square.
 */

#include "SobelSquareImageFilter.h"
#include "./src/common/Constants.h"

SobelSquareImageFilter::SobelSquareImageFilter() {
	Constants *constants = Constants::Instance();

	this->setFilterName(constants->getSobelSquaredConstant());
	this->setFilterMatrixX(constants->getSobelSquareX());
	this->setFilterMatrixY(constants->getSobelSquareY());

}

SobelSquareImageFilter::~SobelSquareImageFilter() {
	// TODO Auto-generated destructor stub
}

