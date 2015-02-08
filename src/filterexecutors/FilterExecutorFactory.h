/*
 * FilterExecutorFactory.h
 *
 *  Created on: 23/03/2012
 *      Author: jose
 */

#ifndef FILTEREXECUTORFACTORY_H_
#define FILTEREXECUTORFACTORY_H_

#include "./src/imagefilters/ImageFilter.h"
#include "FilterExecutor.h"

class FilterExecutorFactory {
public:
	FilterExecutorFactory();
	virtual ~FilterExecutorFactory();
	FilterExecutor *getExecutorInstance(ImageFilter filter);
};

#endif /* FILTEREXECUTORFACTORY_H_ */
