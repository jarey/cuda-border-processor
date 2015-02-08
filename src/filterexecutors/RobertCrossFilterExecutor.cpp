/*
 * RobertCrossFilterExecutor.cpp
 *
 * Creado: 26/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de Robert Cross en CPu, GPu y CPU vd GPU.
 */

#include "RobertCrossFilterExecutor.h"
#include "./src/imagefilters/RobertCrossImageFilter.h"
#include <stdio.h>
#include "./src/imagefilters/ImageAlgorythmBusiness.h"
#include "./src/common/Controlador.h"

RobertCrossFilterExecutor::RobertCrossFilterExecutor() {
	// TODO Auto-generated constructor stub

}

RobertCrossFilterExecutor::~RobertCrossFilterExecutor() {
	// TODO Auto-generated destructor stub
}

QImage RobertCrossFilterExecutor::executeFilterCPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	RobertCrossImageFilter imageFilter = dynamic_cast<RobertCrossImageFilter&>(filter);
	returnImage = imageAlgorythmBusiness->RobertCrossFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~RobertCrossImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage RobertCrossFilterExecutor::executeFilterGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	RobertCrossImageFilter imageFilter = dynamic_cast<RobertCrossImageFilter&>(filter);
	//returnImage = imageAlgorythmBusiness->RobertCrossFilterGPU(image);
	returnImage = imageAlgorythmBusiness->RobertCrossFilterGPUShared(image);
	free(imageAlgorythmBusiness);
	imageFilter.~RobertCrossImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}

QImage RobertCrossFilterExecutor::executeFilterCPUvsGPU(QImage image, ImageFilter &filter){
	Controlador *controlador = Controlador::Instance();
	QImage returnImage;
	ImageAlgorythmBusiness *imageAlgorythmBusiness = new ImageAlgorythmBusiness();
	RobertCrossImageFilter imageFilter = dynamic_cast<RobertCrossImageFilter&>(filter);
	//return imageAlgorythmBusiness->PrewittFilterGPU(image);
	returnImage = imageAlgorythmBusiness->RobertCrossFilterGPUShared(image);
	returnImage = imageAlgorythmBusiness->RobertCrossFilterCPU(image,imageFilter);
	free(imageAlgorythmBusiness);
	imageFilter.~RobertCrossImageFilter();
	controlador->manageExecutionTimeWriting();
	controlador->resetExecutionTimeValues();
	return returnImage;
}
