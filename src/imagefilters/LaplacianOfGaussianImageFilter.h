/*
 * LaplacianOfGaussianImageFilter.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: objeto que representa el filtro de Laplaciano De Gauss.
 */

#ifndef LAPLACIANOFGAUSSIANIMAGEFILTER_H_
#define LAPLACIANOFGAUSSIANIMAGEFILTER_H_

#include "ImageFilter.h"
#include "LaplaceImageFilter.h"

class LaplacianOfGaussianImageFilter : public ImageFilter{
public:
	LaplacianOfGaussianImageFilter();
	LaplacianOfGaussianImageFilter(double radioGauss,double sigmaGauss);
	virtual ~LaplacianOfGaussianImageFilter();

	double getRadioGauss();
	double getSigmaGauss();
	LaplaceImageFilter getLaplaceFilter();

	void setRadioGauss(double radioGauss);
	void setSigmaGauss(double sigmaGauss);
	void setLaplaceFilter(LaplaceImageFilter laplaceFilter);
private:
	LaplaceImageFilter laplaceFilter;
	double radioGauss;
	double sigmaGauss;
};

#endif /* LAPLACIANOFGAUSSIANIMAGEFILTER_H_ */
