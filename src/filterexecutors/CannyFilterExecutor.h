/*
 * CannyFilterExecutor.h
 *
 * Creado: 27/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de canny en CPU, GPU y CPU vs GPU.
 */

#ifndef CANNYFILTEREXECUTOR_H_
#define CANNYFILTEREXECUTOR_H_

#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class CannyFilterExecutor : public FilterExecutor{
public:
	CannyFilterExecutor();
	virtual ~CannyFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* CANNYFILTEREXECUTOR_H_ */
