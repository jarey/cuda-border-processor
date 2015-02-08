/*
 * LaplaceImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Laplace.
 */

#include "LaplaceImageFilter.h"
#include "./src/common/Constants.h"

LaplaceImageFilter::LaplaceImageFilter() {
	Constants *constants = Constants::Instance();

	this->setFilterName(constants->getLaplaceConstant());
	this->setFilterMatrixX(constants->getLaplace());

	FloatMatrix nullMatrix = FloatMatrix(0,0);
	nullMatrix.setElements(0);
	this->setFilterMatrixY(nullMatrix);

}

LaplaceImageFilter::~LaplaceImageFilter() {
	// TODO Auto-generated destructor stub
}
