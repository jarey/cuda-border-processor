/*
 * RobertCrossImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Robert Cross.
 */

#include "RobertCrossImageFilter.h"
#include "./src/common/Constants.h"

RobertCrossImageFilter::RobertCrossImageFilter() {
	// TODO Auto-generated constructor stub
	Constants *constants = Constants::Instance();

	this->setFilterName(constants->getRobertCrossConstant());
	this->setFilterMatrixX(constants->getRobertCrossX());
	this->setFilterMatrixY(constants->getRobertCrossY());

}

RobertCrossImageFilter::~RobertCrossImageFilter() {
	// TODO Auto-generated destructor stub
}



