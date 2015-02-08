/*
 * SobelFilterExecutor.h
 *
 *  Created on: 23/03/2012
 *      Author: jose
 */

#ifndef SOBELFILTEREXECUTOR_H_
#define SOBELFILTEREXECUTOR_H_

#include "FilterExecutor.h"
#include "./src/imagefilters/ImageFilter.h"

class SobelFilterExecutor : public FilterExecutor {
public:
	SobelFilterExecutor();
	virtual ~SobelFilterExecutor();
	QImage executeFilterCPU(QImage image, ImageFilter &filter);
	QImage executeFilterGPU(QImage image, ImageFilter &filter);
	QImage executeFilterCPUvsGPU(QImage image, ImageFilter &filter);
};

#endif /* SOBELFILTEREXECUTOR_H_ */
