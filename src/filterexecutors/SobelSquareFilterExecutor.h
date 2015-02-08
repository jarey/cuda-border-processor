/*
 * SobelSquareFilterExecutor.h
 *
 * Creado: 27/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso SobelSquare en CPU, GPU y CPU vs GPU.
 */

#ifndef SOBELSQUAREFILTEREXECUTOR_H_
#define SOBELSQUAREFILTEREXECUTOR_H_

#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class SobelSquareFilterExecutor : public FilterExecutor {
public:
	SobelSquareFilterExecutor();
	virtual ~SobelSquareFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* SOBELSQUAREFILTEREXECUTOR_H_ */
