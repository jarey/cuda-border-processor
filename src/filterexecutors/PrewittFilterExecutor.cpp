/*
 * PrewittFilterExecutor.cpp
 *
 * Creado: 26/03/2012
 * Author: jose
 * Descripción: ejecutor de los procesos de Prewitt en CPU, GPu y CPU vs GPU.
 */

#include "PrewittFilterExecutor.h"
#include "./src/imagefilters/PrewittImageFilter.h"
#include <stdio.h>
#include <QImage>
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include "./src/common/Controlador.h"

PrewittFilterExecutor::PrewittFilterExecutor() {
	// TODO Auto-generated constructor stub

}

PrewittFilterExecutor::~PrewittFilterExecutor() {
	// TODO Auto-generated destructor stub
}

QImage PrewittFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	PrewittImageFilter imageFilter = dynamic_cast<PrewittImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->PrewittFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~PrewittImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage PrewittFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	PrewittImageFilter imageFilter = dynamic_cast<PrewittImageFilter&>(filter);
	//return imageAlgorythmBusiness->PrewittFilterGPU(image);
	returnImage = imageAlgorythmBusiness->PrewittFilterGPUShared(image);
	free(imageAlgorythmBusiness);
	imageFilter.~PrewittImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage PrewittFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
	//return imageAlgorythmBusiness->PrewittFilterGPU(image);
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	PrewittImageFilter imageFilter = dynamic_cast<PrewittImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->PrewittFilterGPUShared(image);
	returnImage = imageAlgorythmBusiness->PrewittFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~PrewittImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

