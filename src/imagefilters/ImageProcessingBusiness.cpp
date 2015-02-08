/*
 * ImageProcessingBusiness.cpp
 *
 * Creado: 24/03/2012
 * Autor: jose
 * Descripciñon: clase encargada de orquestrar la ejecución de los algoritmos de forma automática.
 * Se hace uso del filtro concreto instanciado por pantalla y se asigna el evaluador de filtro
 * concreto, según el filtro haciendo uso de la factoría de evaluadores.
 */

#include "ImageProcessingBusiness.h"
#include "./src/common/Controlador.h"
#include "./src/common/Constants.h"
#include <QString>
#include <stdio.h>
#include "./src/imagefilters/SobelImageFilter.h"
#include "./src/filterexecutors/FilterExecutor.h"
#include "./src/filterexecutors/FilterExecutorFactory.h"
#include "./src/filterexecutors/SobelFilterExecutor.h"

#include "./src/imagefilters/LaplacianOfGaussianImageFilter.h"
#include <QDebug>
#include <QTime>

ImageProcessingBusiness::ImageProcessingBusiness() {
	// TODO Auto-generated constructor stub

}

ImageProcessingBusiness::~ImageProcessingBusiness() {
	// TODO Auto-generated destructor stub
}

void ImageProcessingBusiness::doProcess(ImageFilter &filter) {
	//Se obtienen los datos del controlador (modo de aplicación)
	Controlador *controlador = Controlador::Instance();
	Constants *constants = Constants::Instance();
	QString algorythmSelected = controlador->getAlgorythmSelected();
	QString applicationMode = controlador->getApplicationMode();
	FilterExecutorFactory *filterExecutorFactory = new FilterExecutorFactory();
	FilterExecutor *filterExecutor =
			filterExecutorFactory->getExecutorInstance(filter);
	QImage returned;

	if (applicationMode == constants->getSimpleImageMode() || applicationMode
			== constants->getPlotMode() || applicationMode
			== constants->getFrameCaptureMode()) {

		if (controlador->getIsGpuMode() == 1) {
			returned = filterExecutor->executeFilterGPU(
					controlador->getMatrixListOrigin()->at(0), filter);
		} else if (controlador->getIsGpuMode() == 0) {
			returned = filterExecutor->executeFilterCPU(
					controlador->getMatrixListOrigin()->at(0), filter);
		} else if (controlador->getIsGpuMode() == 2) {
			//Modo comparación entre CPU y GPU
			returned = filterExecutor->executeFilterCPUvsGPU(
					controlador->getMatrixListOrigin()->at(0), filter);
		}

		controlador->getMatrixListDestiny()->clear();
		controlador->getMatrixListDestiny()->append(returned);
	} else if (applicationMode == constants->getMultipleImageMode()) {

		for (int i = 0; i < controlador->getMatrixListOrigin()->size(); i++) {

			if (controlador->getIsGpuMode() == 1) {
				returned = filterExecutor->executeFilterGPU(
						controlador->getMatrixListOrigin()->at(i), filter);
			} else if (controlador->getIsGpuMode() == 0) {
				returned = filterExecutor->executeFilterCPU(
						controlador->getMatrixListOrigin()->at(i), filter);
			} else if (controlador->getIsGpuMode() == 2) {
				//modo comparación entre CPU y GPU
				returned = filterExecutor->executeFilterCPUvsGPU(
						controlador->getMatrixListOrigin()->at(i), filter);
			}

			if (!returned.isNull()) {
				controlador->getMatrixListDestiny()->append(returned);
			}
		}

	} else if (applicationMode == constants->getMultipleAlgorythmMode()) {

	}

	//Se realiza el borrado de todas las estructuras empleadas.
	free(filterExecutorFactory);
	free(filterExecutor);

}

void ImageProcessingBusiness::doProcess(QList<ImageFilter*> *filters) {
	//Se obtienen los datos del controlador (modo de aplicación)
	Controlador *controlador = Controlador::Instance();
	Constants *constants = Constants::Instance();
	QString algorythmSelected = controlador->getAlgorythmSelected();
	QString applicationMode = controlador->getApplicationMode();

	if (applicationMode == constants->getSimpleImageMode()) {
		//NO SE CONTEMPLA PERO EN CASO DE NECESITARSE IMPLEMENTACIÓN PARA ESTE MODO IRÍA AQUÍ

	} else if (applicationMode == constants->getMultipleImageMode()) {
		//NO SE CONTEMPLA PERO EN CASO DE NECESITARSE IMPLEMENTACIÓN PARA ESTE MODO IRÍA AQUÍ

	} else if (applicationMode == constants->getMultipleAlgorythmMode()) {
		//Se recorren los filtros del listado y para cada uno de ellos se realiza la ejecución del algoritmo correspondiente
		for (int i = 0; i < filters->size(); i++) {
			ImageFilter *filter = new ImageFilter;
			filter = filters->at(i);
			FilterExecutorFactory filterExecutorFactory =
					(FilterExecutorFactory) *new FilterExecutorFactory;
			FilterExecutor* filterExecutor =
					filterExecutorFactory.getExecutorInstance(*filter);
			QImage returned;
			if (controlador->getIsGpuMode() == 1) {
				returned = filterExecutor->executeFilterGPU(
						controlador->getMatrixListOrigin()->at(0), *filter);
			} else if (controlador->getIsGpuMode() == 0) {
				returned = filterExecutor->executeFilterCPU(
						controlador->getMatrixListOrigin()->at(0), *filter);
			} else if (controlador->getIsGpuMode() == 2) {
				//Modo comparación entre CPU y GPU
			}

			if (!returned.isNull()) {
				controlador->getMatrixListDestiny()->append(returned);
			}

		}
	}
}

