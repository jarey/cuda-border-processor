/*
 * TimeContainer.cpp: implementación, de acuerdo al fichero de definición TimeContainer.h
 * 						de la clase de contenedor de tiempos.
 *
 *  Creado: 13/06/2012
 *  Autor: jose
 */

#include "TimeContainer.h"

/*
 * Constructor: TimeContainer
 * Propósito: Realiza la isntanciación d eun objeto contenedor de tiempos con los tiempos
 * 				de CPU y GPU nulos.
 *
 * Creado: 12/07/2012
 * Autor: jose
 *
 */
TimeContainer::TimeContainer() {
	this->GPUmilliseconds = NULL;
	this->CPUmilliseconds = NULL;
}

/*
 * Destructor: TimeContainer
 * Propósito: realiza el borrado de los atributos de la entidad.
 *
 * Creado: 12/07/2012
 * Autor: jose
 *
 */
TimeContainer::~TimeContainer() {
}


//Getters y setters
void TimeContainer::setProcess(QString process){
	this->process = process;
}
void TimeContainer::setCPUMilliseconds(int cpuMillisecods){
	this->CPUmilliseconds = cpuMillisecods;
}

void TimeContainer::setGPUMilliseconds(int gpuMilliseconds){
	this->GPUmilliseconds = gpuMilliseconds;
}

void TimeContainer::setGraphiAceleration(float graphicAceleration){
		this->graphicAceleration = graphicAceleration;
}
void TimeContainer::setMode(QString mode){
	this->mode = mode;
}
QString TimeContainer::getProcess(){
	return this->process;
}
int TimeContainer::getCPUMilliseconds(){
	return this->CPUmilliseconds;
}

int TimeContainer::getGPUMilliseconds(){
		return this->GPUmilliseconds;
}

float TimeContainer::getGraphiAceleration(){
		return this->graphicAceleration;
}
QString TimeContainer::getMode(){
	return this->mode;
}

void TimeContainer::setExecutionType(QString type){
	this->executionType = type;
}
QString TimeContainer::getExecutionType(){
	return this->executionType;
}
