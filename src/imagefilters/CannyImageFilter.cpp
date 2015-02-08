/*
 * CannyImageFilter.cpp
 *
 * Creado: 29/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de canny.
 */

#include "CannyImageFilter.h"
#include "./src/common/Constants.h"

CannyImageFilter::CannyImageFilter() {
	Constants *constants = Constants::Instance();
	this->setFilterName(constants->getCannyConstant());
}

CannyImageFilter::CannyImageFilter(bool histeresis,float radioGauss,float sigmaGauss, float lowerThreshold, float higherThreshold){
	Constants* constants = Constants::Instance();
	this->setFilterName(constants->getCannyConstant());
	this->setHisteresis(histeresis);
	this->setRadioGauss(radioGauss);
	this->setSigmaGauss(sigmaGauss);
	this->setLowerThreshold(lowerThreshold);
	this->setHigherThreshold(higherThreshold);

	//Se componen la matriz de gauss y se setea al filtro.
	FloatMatrix gaussMatrix = FloatMatrix(radioGauss,radioGauss);
	//LLamada al método de generación de la matriz.
	this->setGaussMatrix(gaussMatrix);
}

CannyImageFilter::~CannyImageFilter() {
	// TODO Auto-generated destructor stub
}

//Getters
bool CannyImageFilter::getHisteresis() {
	return this->histeresis;
}
float CannyImageFilter::getRadioGauss() {
	return this->radioGauss;
}
float CannyImageFilter::getSigmaGauss() {
	return this->sigmaGauss;
}
float CannyImageFilter::getLowerThreshold() {
	return this->lowerThreshold;
}
float CannyImageFilter::getHigherThreshold() {
	return this->higherThreshold;
}
FloatMatrix CannyImageFilter::getGaussMatrix(FloatMatrix gaussMatrix) {
	return this->gaussMatrix;
}
//Setters
void CannyImageFilter::setHisteresis(bool histeresis) {
	this->histeresis = histeresis;
}
void CannyImageFilter::setRadioGauss(float radioGauss) {
	this->radioGauss = radioGauss;
}
void CannyImageFilter::setSigmaGauss(float sigmaGauss) {
	this->sigmaGauss = sigmaGauss;
}
void CannyImageFilter::setLowerThreshold(float lowerThreshold) {
	this->lowerThreshold = lowerThreshold;
}
void CannyImageFilter::setHigherThreshold(float higherThreshold) {
	this->higherThreshold = higherThreshold;
}
void CannyImageFilter::setGaussMatrix(FloatMatrix gaussMatrix) {
	this->gaussMatrix = gaussMatrix;
}
