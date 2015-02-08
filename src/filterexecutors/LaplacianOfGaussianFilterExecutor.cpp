/*
 * LaplacianOfGaussianFilterExecutor.cpp
 *
 * Creado: 26/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso Laplaciano De Gauss en CPu, GPU y CPU vs GPU.
 */

#include "LaplacianOfGaussianFilterExecutor.h"
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include "./src/imagefilters/LaplacianOfGaussianImageFilter.h"
#include <QDebug>
#include "./src/common/Controlador.h"

LaplacianOfGaussianFilterExecutor::LaplacianOfGaussianFilterExecutor() {
	// TODO Auto-generated constructor stub
}

LaplacianOfGaussianFilterExecutor::~LaplacianOfGaussianFilterExecutor() {
	// TODO Auto-generated destructor stub
}


QImage LaplacianOfGaussianFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	LaplacianOfGaussianImageFilter imageFilter = dynamic_cast<LaplacianOfGaussianImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->LaplacianOfGaussianFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage LaplacianOfGaussianFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	LaplacianOfGaussianImageFilter imageFilter = dynamic_cast<LaplacianOfGaussianImageFilter&>(filter);
	//returnImage = imageAlgorythmBusiness->LaplacianOfGaussianFilterGPU(image,imageFilter);
	returnImage = imageAlgorythmBusiness->LaplacianOfGaussianFilterGPUShared(image,imageFilter);
	free(imageAlgorythmBusiness);
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}


QImage LaplacianOfGaussianFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	LaplacianOfGaussianImageFilter imageFilter = dynamic_cast<LaplacianOfGaussianImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->LaplacianOfGaussianFilterGPUShared(image,imageFilter);
	returnImage = imageAlgorythmBusiness->LaplacianOfGaussianFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	//Gestión de tiempos de ejecución
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;

}
