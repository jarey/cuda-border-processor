/*
 * CannyFilterExecutor.cpp
 *
 * Creado: 27/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de canny en CPU, GPU y CPU vs GPU.
 */

#include "CannyFilterExecutor.h"
#include "./src/imagefilters/CannyImageFilter.h"
#include <stdio.h>
#include <QImage>
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include "./src/common/Controlador.h"

CannyFilterExecutor::CannyFilterExecutor() {
	// TODO Auto-generated constructor stub

}

CannyFilterExecutor::~CannyFilterExecutor() {
	// TODO Auto-generated destructor stub
}

QImage CannyFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){

	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	CannyImageFilter imageFilter = dynamic_cast<CannyImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->CannyFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;

}

QImage CannyFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	CannyImageFilter imageFilter = dynamic_cast<CannyImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->cannyGpuShared(image,imageFilter);
	free(imageAlgorythmBusiness);
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage CannyFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	CannyImageFilter imageFilter = dynamic_cast<CannyImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->cannyGpuShared(image,imageFilter);
	returnImage = imageAlgorythmBusiness->CannyFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

