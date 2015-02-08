/*
 * SobelSquareFilterExecutor.cpp
 *
 * Creado: 27/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso SobelSquare en CPU, GPU y CPU vs GPU.
 */

#include "SobelSquareFilterExecutor.h"
#include "./src/imagefilters/SobelSquareImageFilter.h"
#include <stdio.h>
#include <QImage>
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include "./src/common/Controlador.h"

SobelSquareFilterExecutor::SobelSquareFilterExecutor() {
	// TODO Auto-generated constructor stub

}

SobelSquareFilterExecutor::~SobelSquareFilterExecutor() {
	// TODO Auto-generated destructor stub
}

QImage SobelSquareFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	SobelSquareImageFilter imageFilter = dynamic_cast<SobelSquareImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->SobelSquareFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~SobelSquareImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage SobelSquareFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	SobelSquareImageFilter imageFilter = dynamic_cast<SobelSquareImageFilter&>(filter);
	//return imageAlgorythmBusiness->SobelSquareFilterGPU(image);
	returnImage = imageAlgorythmBusiness->SobelSquareFilterGPUShared(image);
	free(imageAlgorythmBusiness);
	imageFilter.~SobelSquareImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}


QImage SobelSquareFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
//	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
//	SobelSquareImageFilter imageFilter = dynamic_cast<SobelSquareImageFilter&>(filter);
//	//return imageAlgorythmBusiness->SobelSquareFilterGPU(image);
//	return imageAlgorythmBusiness->SobelSquareFilterGPUShared(image);

	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	SobelSquareImageFilter imageFilter = dynamic_cast<SobelSquareImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->SobelSquareFilterGPUShared(image);
	returnImage = imageAlgorythmBusiness->SobelSquareFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);

	//Gestión de tiempos de ejecución
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();

	return returnImage;
}
