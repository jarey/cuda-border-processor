/*
 * FilterExecutor.h
 *
 *  Created on: 23/03/2012
 *      Author: jose
 */

#ifndef FILTEREXECUTOR_H_
#define FILTEREXECUTOR_H_

#include <QImage>
#include "./src/imagefilters/ImageFilter.h"

class FilterExecutor {
public:
	FilterExecutor();
	virtual ~FilterExecutor();
	QImage virtual executeFilterCPU(QImage image, ImageFilter &filter)=0;
	QImage virtual executeFilterGPU(QImage image, ImageFilter &filter)=0;
	QImage virtual executeFilterCPUvsGPU(QImage image, ImageFilter &filter)=0;
};

#endif /* FILTEREXECUTOR_H_ */
