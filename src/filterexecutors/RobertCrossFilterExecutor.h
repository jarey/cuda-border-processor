/*
 * RobertCrossFilterExecutor.h
 *
 * Creado: 26/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de Robert Cross en CPu, GPu y CPU vd GPU.
 */

#ifndef ROBERTCROSSFILTEREXECUTOR_H_
#define ROBERTCROSSFILTEREXECUTOR_H_

#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class RobertCrossFilterExecutor : public FilterExecutor {
public:
	RobertCrossFilterExecutor();
	virtual ~RobertCrossFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* ROBERTCROSSFILTEREXECUTOR_H_ */
