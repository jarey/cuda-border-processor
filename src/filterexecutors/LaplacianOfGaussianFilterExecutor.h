/*
 * LaplacianOfGaussianFilterExecutor.h
 *
 * Creado: 26/03/2012
 * Autor: jose
 * Descripción: ejecutor del proceso Laplaciano De Gauss en CPu, GPU y CPU vs GPU.
 */

#ifndef LAPLACIANOFGAUSSIANFILTEREXECUTOR_H_
#define LAPLACIANOFGAUSSIANFILTEREXECUTOR_H_

#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class LaplacianOfGaussianFilterExecutor : public FilterExecutor{
public:
	LaplacianOfGaussianFilterExecutor();
	virtual ~LaplacianOfGaussianFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* LAPLACIANOFGAUSSIANFILTEREXECUTOR_H_ */
