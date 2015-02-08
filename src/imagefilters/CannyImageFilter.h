/*
 * CannyImageFilter.h
 *
 * Creado: 29/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de canny.
 */

#ifndef CANNYIMAGEFILTER_H_
#define CANNYIMAGEFILTER_H_

#include "ImageFilter.h"
#include "./src/common/FloatMatrix.h"

class CannyImageFilter :public ImageFilter {
public:
	CannyImageFilter();
	virtual ~CannyImageFilter();
	CannyImageFilter(bool histeresis,float radioGauss,float sigmaGauss, float lowerThreshold, float higherThreshold);
	//Setters
	bool getHisteresis();
	float getRadioGauss();
	float getSigmaGauss();
	float getLowerThreshold();
	float getHigherThreshold();
	FloatMatrix getGaussMatrix(FloatMatrix gaussMatrix);
	//Getters
	void setHisteresis(bool histeresis);
	void setRadioGauss(float radioGauss);
	void setSigmaGauss(float sigmaGauss);
	void setLowerThreshold(float lowerThreshold);
	void setHigherThreshold(float higherThreshold);
	void setGaussMatrix(FloatMatrix gaussMatrix);

private:
	//Atributos específicos del filtro de canny que no tiene en herencia.
	//matriz de gaus compuesta en el constructor a partir de los parámetros de gauss
	FloatMatrix gaussMatrix;
	//Histeresis
	bool histeresis;
	//radios gauss
	float radioGauss;
	//sigma gauss
	float sigmaGauss;
	//valor umbral inferior
	float lowerThreshold;
	//valor umbral superior
	float higherThreshold;
};

#endif /* CANNYIMAGEFILTER_H_ */
