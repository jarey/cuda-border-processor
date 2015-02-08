/*
 * LaplacianOfGaussianImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Laplaciano De Gauss.
 */

#include "LaplacianOfGaussianImageFilter.h"
#include "./src/common/Constants.h"

#include <QDebug>

LaplacianOfGaussianImageFilter::LaplacianOfGaussianImageFilter() {
	Constants *constants = Constants::Instance();
	this->setFilterName(constants->getLaplacianOfGaussianConstant());
}


LaplacianOfGaussianImageFilter::LaplacianOfGaussianImageFilter(double radioGauss,double sigmaGauss) {
	Constants *constants = Constants::Instance();

	this->setRadioGauss(radioGauss);
	this->setSigmaGauss(sigmaGauss);
	qDebug() << radioGauss;
	this->setFilterName(constants->getLaplacianOfGaussianConstant());
	LaplaceImageFilter laplaceFilter = LaplaceImageFilter();
	this->setLaplaceFilter(laplaceFilter);
}

LaplacianOfGaussianImageFilter::~LaplacianOfGaussianImageFilter() {
	// TODO Auto-generated destructor stub
}

//Getters

double LaplacianOfGaussianImageFilter::getRadioGauss() {
	return this->radioGauss;
}
double LaplacianOfGaussianImageFilter::getSigmaGauss() {
	return this->sigmaGauss;
}
LaplaceImageFilter LaplacianOfGaussianImageFilter::getLaplaceFilter(){
	return this->laplaceFilter;
}

//Setters

void LaplacianOfGaussianImageFilter::setRadioGauss(double radioGauss) {
	this->radioGauss = radioGauss;
}
void LaplacianOfGaussianImageFilter::setSigmaGauss(double sigmaGauss) {
	this->sigmaGauss = sigmaGauss;
}
void LaplacianOfGaussianImageFilter::setLaplaceFilter(LaplaceImageFilter laplaceFiltare){
	this->laplaceFilter = laplaceFiltare;
}


