/*
 * SobelFilterExecutor.cpp
 *
 * Creado: 23/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de Sobel en CPU, GPU y CPU vs GPU.
 */

#include "SobelFilterExecutor.h"
#include "./src/imagefilters/SobelImageFilter.h"
#include <stdio.h>
#include <QImage>
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include <QDebug>
#include "./src/common/Controlador.h"

SobelFilterExecutor::SobelFilterExecutor() {
	// TODO Auto-generated constructor stub

}

SobelFilterExecutor::~SobelFilterExecutor() {
	// TODO Auto-generated destructor stub

}

QImage SobelFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	SobelImageFilter imageFilter = dynamic_cast<SobelImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->SobelFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~SobelImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage SobelFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	//returnImage = imageAlgorythmBusiness->SobelFilterGPU(image);
	returnImage = imageAlgorythmBusiness->SobelFilterGPUShared(image);
	free(imageAlgorythmBusiness);
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}


QImage SobelFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	SobelImageFilter imageFilter = dynamic_cast<SobelImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->SobelFilterGPUShared(image);
	returnImage = imageAlgorythmBusiness->SobelFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	//Gestión de tiempos de ejecución
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}
