/*
 * Controlador.h
 *
 *  Created on: 21/03/2012
 *      Author: jose
 */

#ifndef CONTROLADOR_H_
#define CONTROLADOR_H_

#include <QList>
#include <QImage>
#include <QString>
#include <QStringList>
#include "./src/common/TimeContainer.h"

class Controlador {
public:
	static Controlador* Instance();
	//Getters
	QImage getOriginImage();
	QImage getResultImage();
	QList<QImage>* getMatrixListOrigin();
	QList<QImage>* getMatrixListDestiny();
	QString getApplicationMode();
	QString getAlgorythmSelected();
	unsigned char getIsGpuMode();
	unsigned char getIsCudaCapable();
	double getCpuExecutionTime();
	double getGpuExecutionTime();

	//Setters
	void setOriginImage(QImage originImage);
	void setResultImage(QImage resultImage);
	void setMatrixListOrigin(QList<QImage>* matrixListOrigin);
	void setMatrixListDestiny(QList<QImage>* matrixListDestiny);
	void setApplicationMode(QString applicationMode);
	void setAlgorythmSelected(QString algorythmSelected);
	void setIsGpuMode(unsigned char value);
	void setIsCudaCapable(unsigned char value);
	void setCpuExecutionTime(double cpuTime);
	void setGpuExecutionTime(double gpuTime);

	//Getters y setters para los mensajes de tiempo de ejecución de los algoritmos según el modo
	void setSimpleImageExecution(QList<TimeContainer>* simpleImage);
	void setMultipleImageExecution(QList<TimeContainer>* multipleImage );
	void setMultipleAlgorythmExecution(QList<TimeContainer>* multipleAlgorythm);
	void setGraphMode(QList<TimeContainer>* graph);
	//Listado de tiempos de ejecución para cada algoritmo.
	QList<TimeContainer>* getSimpleImageExecution();
	QList<TimeContainer>* getMultipleImageExecution();
	QList<TimeContainer>* getMultipleAlgorythmExecution();
	QList<TimeContainer>* getGraphMode();
	//Limpiado de imágenes resultado y origen.
	void cleanResultList();
	void cleanOriginList();
	//Borrado de información de tiempos en el
	void resetExecutionTimeValues();
	//Tranferencia de imágenes entre listados.
	void fromQlist(QList<QImage>* matrixList);
	//Gestión de información de tiempos.
	void manageExecutionTimeWriting();
	void setTimeAlgorythm(QString algorythm);
	QString getTimeAlgorythm();


protected:
	//Constructores privador para implementación de patrón singleton
	Controlador();
	Controlador(const Controlador & ) ;
	Controlador &operator= (const Controlador & ) ;

private:
	static Controlador* pinstance;
	QImage originImage;
	QImage resultImage;
	QList<QImage> *matrixListOrigin;
	QList<QImage> *matrixListDestiny;
	QString applicationMode;
	QString algorythmSelected;
	unsigned char isGpuEnable;
	unsigned char isCudaCapable;

	//Listados de mensajes de ejecución de tiempos para cada
	QList<TimeContainer> *simpleImageExecution;
	QList<TimeContainer> *multipleImageExecution;
	QList<TimeContainer> *multipleAlgorythmExecution;
	QList<TimeContainer> *graphMode;
	double cpuExecutionTime;
	double gpuExecutionTime;
	//Tiempo de ejecución del algoritmo.
	QString timeAlgorythm;
};

#endif /* CONTROLADOR_H_ */

