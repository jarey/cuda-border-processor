/*
 * FilterExecutorFactory.cpp
 *
 *  Created on: 23/03/2012
 *      Author: jose
 */

#include "FilterExecutorFactory.h"
#include "SobelFilterExecutor.h"
#include "LaplaceFilterExecutor.h"
#include "PrewittFilterExecutor.h"
#include "RobertCrossFilterExecutor.h"
#include "LaplacianOfGaussianFilterExecutor.h"
#include "SobelSquareFilterExecutor.h"
#include "CannyFilterExecutor.h"
#include "./src/common/Constants.h"

FilterExecutorFactory::FilterExecutorFactory() {
	// TODO Auto-generated constructor stub

}

FilterExecutorFactory::~FilterExecutorFactory() {
	// TODO Auto-generated destructor stub
}

FilterExecutor *FilterExecutorFactory::getExecutorInstance(ImageFilter filter) {
	Constants *constants = Constants::Instance();

	if(filter.getFilterName() == constants->getSobelConstant()) {
		return new SobelFilterExecutor;
	}else if(filter.getFilterName() == constants->getPrewittConstant()){
		return new PrewittFilterExecutor;
	}else if(filter.getFilterName() == constants->getRobertCrossConstant()){
		return new RobertCrossFilterExecutor;
	}else if(filter.getFilterName() == constants->getLaplaceConstant()){
		return new LaplaceFilterExecutor;
	}else if(filter.getFilterName() == constants->getLaplacianOfGaussianConstant()){
		return new LaplacianOfGaussianFilterExecutor;
	}else if(filter.getFilterName() == constants->getSobelSquaredConstant()){
		return new SobelSquareFilterExecutor;
	}else if(filter.getFilterName() == constants->getCannyConstant()){
		return new CannyFilterExecutor;
	}
}
