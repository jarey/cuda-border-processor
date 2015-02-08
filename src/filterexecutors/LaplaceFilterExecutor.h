/*
 * LaplaceFilterExecutor.h
 *
 * Creado: 26/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso de Laplace en CPu, GPU y CPU vs GPU.
 */

#ifndef LAPLACEFILTEREXECUTOR_H_
#define LAPLACEFILTEREXECUTOR_H_

#include <QImage>
#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class LaplaceFilterExecutor : public FilterExecutor {
public:
	LaplaceFilterExecutor();
	virtual ~LaplaceFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* LAPLACEFILTEREXECUTOR_H_ */
