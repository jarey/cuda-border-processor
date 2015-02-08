/*
 * SobelFilterExecutor.h
 *
 * Creado: 23/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de Sobel en CPU, GPU y CPU vs GPU.
 */

#ifndef PREWITTFILTEREXECUTOR_H_
#define PREWITTFILTEREXECUTOR_H_

#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class PrewittFilterExecutor : public FilterExecutor{
public:
	PrewittFilterExecutor();
	virtual ~PrewittFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* PREWITTFILTEREXECUTOR_H_ */
