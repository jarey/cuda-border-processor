/*
 * Controlador.cpp
 * Propósito: clase singleton encargada de realizar procesos globales entre capas y de realizar el control de la aplicación.
 * 			  Esta clase alberga los datos de control y de ejecución de cada una de las funcionalidades de la aplicación, de forma
 * 			  que el flujo de ejecución sigue un orden y un control establecido.
 *
 *  Creado: 21/03/2012
 *  @author: jose
 */

#include "Controlador.h"
#include "Constants.h"

Controlador* Controlador::pinstance = 0;// Inicializar el puntero

/*
 * Método: Instance
 * Propósito: Obtener la instancia de la clase controlador implementación del patrón singleton.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
Controlador* Controlador::Instance() {
	if (pinstance == 0) // ¿Es la primera llamada?
	{
		pinstance = new Controlador; // Creamos la instancia
	}
	return pinstance; // Retornamos la dirección de la instancia
}

/*
 * Constructor: Controlador
 * Propósito: Constructor de la clase, que inicializa por primera vez los valores del flujo de ejecución, estos se utilizan en la apertura de la pantalla
 * 			  para que la aplicación se encuadre de acuerdo al primer panel que se visualiza en la aplicación, en este caso el panel de ejecución simple.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
Controlador::Controlador() {
	Constants *constants = Constants::Instance();

	this->applicationMode = constants->getSimpleImageMode();
	this->algorythmSelected = constants->getCannyConstant();
	this->isGpuEnable = 0;
	this->matrixListDestiny = new QList<QImage> ();
	this->matrixListOrigin = new QList<QImage> ();
	//Listados de mensajes de ejecución de tiempos para cada modo de ejecución.
	this->simpleImageExecution = new QList<TimeContainer> ();
	this->multipleImageExecution = new QList<TimeContainer> ();
	this->multipleAlgorythmExecution = new QList<TimeContainer> ();
	this->graphMode = new QList<TimeContainer> ();
	this->cpuExecutionTime = NULL;
	this->gpuExecutionTime = NULL;
}

//Getters para los atributos del controlador pantalla/lógica

/**
 * Devuelve la imagen original en el caso de los modos de funcionamiento simples
 * **/
QImage Controlador::getOriginImage() {
	return this->originImage;
}

/**
 * Devuelve la resultado del filtrado en el caso de los modos de funcionamiento simples
 * **/

QImage Controlador::getResultImage() {
	return this->resultImage;
}

/**
 * Devuelve el listado de imágenes seleccionadas para los modos de selección múltiple
 * **/
QList<QImage>* Controlador::getMatrixListOrigin() {
	return this->matrixListOrigin;
}

/**
 * Devuelve el listado de imágenes resultado del filtrado para los modos de selección múltiple
 * **/
QList<QImage>* Controlador::getMatrixListDestiny() {
	return this->matrixListDestiny;
}

/**
 * Devuelve el modo de la aplicación en el que se está actualmente
 * **/

QString Controlador::getApplicationMode() {
	return this->applicationMode;
}

/**
 * Devuelve el tipo de algoritmo seleccionado en caso de haber uno
 * **/

QString Controlador::getAlgorythmSelected() {
	return this->algorythmSelected;
}

/**
 * Devuelve el indicador de si se ha seleccionado la ejecución en GPU
 */
unsigned char Controlador::getIsGpuMode() {
	return this->isGpuEnable;
}

/**
 * Devuelve el indicador de si se ha encontrado alguna tarjeta en el equipo compatible con la ejecución en CUDA.
 */
unsigned char Controlador::getIsCudaCapable() {
	return this->isCudaCapable;
}

//Setters
void Controlador::setOriginImage(QImage originImage) {
	this->originImage = originImage;
}
void Controlador::setResultImage(QImage resultImage) {
	this->resultImage = resultImage;
}
void Controlador::setMatrixListOrigin(QList<QImage>* matrixListOrigin) {
	this->matrixListOrigin = matrixListOrigin;
}
void Controlador::setMatrixListDestiny(QList<QImage>* matrixListDestiny) {
	this->matrixListDestiny = matrixListDestiny;
}
void Controlador::setApplicationMode(QString applicationMode) {
	this->applicationMode = applicationMode;
}
void Controlador::setAlgorythmSelected(QString algorythmSelected) {
	this->algorythmSelected = algorythmSelected;
}

void Controlador::setIsGpuMode(unsigned char value) {
	this->isGpuEnable = value;
}
void Controlador::setIsCudaCapable(unsigned char value) {
	this->isCudaCapable = value;
}
void Controlador::setTimeAlgorythm(QString algorythm) {
	this->timeAlgorythm = algorythm;
}
void Controlador::setCpuExecutionTime(double cpuTime) {
	this->cpuExecutionTime = cpuTime;
}
void Controlador::setGpuExecutionTime(double gpuTime) {
	this->gpuExecutionTime = gpuTime;
}
//Getters y setters para los mensajes de tiempo de ejecución de los algoritmos según el modo
void Controlador::setSimpleImageExecution(QList<TimeContainer>* simpleImage) {
	this->simpleImageExecution = simpleImage;
}
void Controlador::setMultipleImageExecution(QList<TimeContainer>* multipleImage) {
	this->multipleImageExecution = multipleImage;
}
void Controlador::setMultipleAlgorythmExecution(
		QList<TimeContainer>* multipleAlgorythm) {
	this->multipleAlgorythmExecution = multipleAlgorythm;
}
void Controlador::setGraphMode(QList<TimeContainer>* graph) {
	this->graphMode = graph;
}
QList<TimeContainer>* Controlador::getSimpleImageExecution() {
	return this->simpleImageExecution;
}
QList<TimeContainer>* Controlador::getMultipleImageExecution() {
	return this->multipleImageExecution;
}
QList<TimeContainer>* Controlador::getMultipleAlgorythmExecution() {
	return this->multipleAlgorythmExecution;
}
QList<TimeContainer>* Controlador::getGraphMode() {
	return this->graphMode;
}
QString Controlador::getTimeAlgorythm() {
	return this->timeAlgorythm;
}
double Controlador::getCpuExecutionTime() {
	return this->cpuExecutionTime;
}
double Controlador::getGpuExecutionTime() {
	return this->gpuExecutionTime;
}

/*
 * Método: cleanResultList
 * Propósito: Realiza el limpiado del listado de imágenes resultado almacenado en el controlador.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
void Controlador::cleanResultList() {
	this->matrixListDestiny->clear();
}

/*
 * Método: cleanOriginList
 * Propósito: Realiza el limpieado de las imágenes origen alojadas en el controlador.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
void Controlador::cleanOriginList() {
	free(this->matrixListOrigin);
	this->matrixListOrigin = new QList<QImage> ();
}

/*
 * Método: fromQlist
 * Propósito: Realiza la información del listado de imágenes origen a partir de un listado de imágenes determinado.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
void Controlador::fromQlist(QList<QImage>* matrixList) {

	this->getMatrixListOrigin()->clear();

	for (int h = 0; h < matrixList->size(); h++) {
		this->getMatrixListOrigin()->append(matrixList->value(h));
	}
}

/*
 * Método: resetExecutionTimeValues
 * Propósito: Informa a NULL los valores de tiempo de ejecución en CPU y en GPU para dejar limpios los valores apra una próxima ejecución.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
void Controlador::resetExecutionTimeValues() {
	this->cpuExecutionTime = NULL;
	this->gpuExecutionTime = NULL;
}

/*
 * Constructor: manageExecutionTimeWriting
 * Propósito: Realiza la gestión de los tiempos de ejecución de los algoritmos del sistema a través de los tiempos de ejecución informados en el controlador
 * 			  informándolos en el listado del modo de ejecución correcto y haciendo las cunetas de aceleración en GPU con respecto a CPU si fuera
 * 			  necesario.
 *
 *  Created on: 21/03/2012
 * @author: jose
 */
void Controlador::manageExecutionTimeWriting() {
	Constants *constants = Constants::Instance();
	TimeContainer container;

	container.setProcess(this->algorythmSelected);
	container.setExecutionType(this->applicationMode);

	//Ejecución CPU
	if (this->getIsGpuMode() == 0) {
		container.setMode("CPU");
		container.setCPUMilliseconds(this->getCpuExecutionTime());
		//Ejecución GPU
	} else if (this->getIsGpuMode() == 1) {
		container.setMode("GPU");
		container.setGPUMilliseconds(this->getGpuExecutionTime());
		//Ejecución CPU vs CPU
	} else if (this->getIsGpuMode() == 2) {
		container.setMode("CPU vs GPU");

		container.setCPUMilliseconds(this->getCpuExecutionTime());
		container.setGPUMilliseconds(this->getGpuExecutionTime());

		if (container.getCPUMilliseconds() != 0
				&& container.getGPUMilliseconds() != 0
				&& container.getCPUMilliseconds() != NULL
				&& container.getGPUMilliseconds() != NULL) {
			//Se han informado el valor de tiempo CPU y tiempo GPU entonce se calcula la aceleracion gráfica
			container.setGraphiAceleration(
					(float) container.getCPUMilliseconds()
							/ (float) container.getGPUMilliseconds());

		}
	}

	if (this->getApplicationMode() == constants->getSimpleImageMode()) {
		this->getSimpleImageExecution()->append(container);
	} else if (this->getApplicationMode() == constants->getMultipleImageMode()) {
		this->getMultipleImageExecution()->append(container);
	} else if (this->getApplicationMode()
			== constants->getMultipleAlgorythmMode()) {
		this->getMultipleAlgorythmExecution()->append(container);
	}//Falta por tener en cuenta el modo gráfica para la aplicación. (se pospone)
}
