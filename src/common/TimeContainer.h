/*
 * TimeContainer.h: Clase de definición que representa un contenedor de tiempos de ejecución.
 * 					Se empleará para encapsular las medidas de tiempo tomadas sobre cada proceso.
 *
 *  Creado: 13/06/2012
 *  Autor: jose
 */

#ifndef TIMECONTAINER_H_
#define TIMECONTAINER_H_


#include <QString>

class TimeContainer {
public:
	//Constructor y destructor.
	TimeContainer();
	virtual ~TimeContainer();
	//Getters y setters
	void setProcess(QString process);
	void setCPUMilliseconds(int cpuMillisecods);
	void setGPUMilliseconds(int gpuMilliseconds);
	void setGraphiAceleration(float graphicAceleration);
	void setMode(QString mode);
	QString getProcess();
	int getCPUMilliseconds();
	int getGPUMilliseconds();
	float getGraphiAceleration();
	QString getMode();
	void setExecutionType(QString type);
	QString getExecutionType();
private:
	//Atributos privados.
	QString process;
	double CPUmilliseconds;
	double GPUmilliseconds;
	float graphicAceleration;
	QString mode;
	QString executionType;
};

#endif /* TIMECONTAINER_H_ */
