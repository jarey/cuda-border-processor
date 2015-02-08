/*
 * LaplaceFilterExecutor.cpp
 *
 * Creado: 26/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de Laplace en CPu, GPU y CPU vs GPU.
 */

#include "LaplaceFilterExecutor.h"
#include "./src/imagefilters/LaplaceImageFilter.h"
#include <stdio.h>
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include <QDebug>
#include "./src/common/Controlador.h"

LaplaceFilterExecutor::LaplaceFilterExecutor() {
	// TODO Auto-generated constructor stub

}

LaplaceFilterExecutor::~LaplaceFilterExecutor() {
	// TODO Auto-generated destructor stub
}

QImage LaplaceFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	LaplaceImageFilter imageFilter = dynamic_cast<LaplaceImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->LaplaceFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~LaplaceImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage LaplaceFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	LaplaceImageFilter imageFilter = dynamic_cast<LaplaceImageFilter&>(filter);
	//returnImage = imageAlgorythmBusiness->LaplaceFilterGPU(image);
	returnImage = imageAlgorythmBusiness->LaplaceFilterGPUShared(image);
	free(imageAlgorythmBusiness);
	imageFilter.~LaplaceImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage LaplaceFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	LaplaceImageFilter imageFilter = dynamic_cast<LaplaceImageFilter&>(filter);
	//return imageAlgorythmBusiness->PrewittFilterGPU(image);
	returnImage = imageAlgorythmBusiness->LaplaceFilterGPUShared(image);
	returnImage = imageAlgorythmBusiness->LaplaceFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~LaplaceImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}
