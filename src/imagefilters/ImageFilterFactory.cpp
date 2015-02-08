/*
 * ImageFilterFactory.cpp
 *
 * Creado: 28/03/2012
 * Autor: jose
 * Descripción: factoría de filtros, se emplean apra obtener de forma automática, el filtro asociado
 * a un proceso escogido por pantalla.
 */

#include "ImageFilterFactory.h"
#include "SobelImageFilter.h"
#include "SobelSquareImageFilter.h"
#include "PrewittImageFilter.h"
#include "LaplaceImageFilter.h"
#include "LaplacianOfGaussianImageFilter.h"
#include "CannyImageFilter.h"
#include "RobertCrossImageFilter.h"
#include "./src/common/Constants.h"

#include <QDebug>

ImageFilterFactory::ImageFilterFactory() {
	// TODO Auto-generated constructor stub

}

ImageFilterFactory::~ImageFilterFactory() {
	// TODO Auto-generated destructor stub
}


ImageFilter *ImageFilterFactory::getImageFilterInstance(QString filterName,bool histeresis, float radioGauss,
		double sigmaGauss, double lowerThreshold, double higherThreshold) {
	Constants *constants = Constants::Instance();

	if (filterName == constants->getSobelConstant()) {
		return new SobelImageFilter();
	} else if (filterName == constants->getSobelSquaredConstant()) {
		return new SobelSquareImageFilter();
	} else if (filterName == constants->getPrewittConstant()) {
		return new PrewittImageFilter();
	} else if (filterName == constants->getRobertCrossConstant()) {
		return new RobertCrossImageFilter();
	} else if (filterName == constants->getLaplaceConstant()) {
		return new LaplaceImageFilter();
	} else if (filterName == constants->getLaplacianOfGaussianConstant()) {
		return new LaplacianOfGaussianImageFilter(radioGauss,sigmaGauss);
	} else if (filterName == constants->getCannyConstant()) {
		return new CannyImageFilter(histeresis, radioGauss, sigmaGauss, lowerThreshold,higherThreshold);
	}
}
