/*
 * Constants.h
 * Propósito: clase que almacena los valores de las constantes empleadas en el programa, constantes para procesos, modos de ejecución
 *			  matrices de filtros y mensajes de ayuda de cada pantalla.
 *
 *  Creado: 21/03/2012
 *      Author: jose
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <QString>
#include "./src/common/FloatMatrix.h"


class Constants {
public:
	//Variable de la propia clase para implementación de patrón singleton.
	static Constants* Instance();
	//Variables para las constantes de los procesos.
	QString getSobelConstant();
	QString getPrewittConstant();
	QString getCannyConstant();
	QString getSobelSquaredConstant();
	QString getLaplaceConstant();
	QString getRobertCrossConstant();
	QString getLaplacianOfGaussianConstant();
	//Variables para constantes de modos de ejecución.
	QString getSimpleImageMode();
	QString getMultipleImageMode();
	QString getMultipleAlgorythmMode();
	QString getPlotMode();
	QString getFrameCaptureMode();

	//Getters para mensajes de ayuda de cada panel
	QString getPanel1HelpMessage();
	QString getPanel2HelpMessage();
	QString getPanel3HelpMessage();
	QString getPanel4HelpMessage();
	QString getPanel5HelpMessage();
	QString getPanel6HelpMessage();
	QString getCzdiaglogHelpMessage();

	//Matrices para cada uno de los filtros
	FloatMatrix getSobelX();
	FloatMatrix getSobelY();
	FloatMatrix getSobelSquareX();
	FloatMatrix getSobelSquareY();
	FloatMatrix getPrewittX();
	FloatMatrix getPrewittY();
	FloatMatrix getRobertCrossX();
	FloatMatrix getRobertCrossY();
	FloatMatrix getLaplace();

	//Setters para las constantes de ayuda de cada algoritmo.
	QString getCannyHelpMessage();
	QString getLaplacianOfGaussianHelpMessage();
	QString getLaplaceHelpMessage();
	QString getSobelHelpMessage();
	QString getSobelSquareHelpMessage();
	QString getRobertCrossHelpMessage();
	QString getPrewittHelpMessage();

protected:
	Constants();
	Constants(const Constants & ) ;
	Constants &operator= (const Constants & ) ;
private:
	static Constants* pinstance;
	//Constantes para tipos de filtros
	QString sobelConstant;
	QString prewittConstant;
	QString cannyConstant;
	QString sobelSquaredConstant;
	QString laplaceConstant;
	QString robertCrossConstant;
	QString laplacianOfGaussianConstant;

	//Constantes para mensajes de ayuda para cada panel
	QString panel1HelpMessage;
	QString panel2HelpMessage;
	QString panel3HelpMessage;
	QString panel4HelpMessage;
	QString panel5HelpMessage;
	QString panel6HelpMessage;
	QString czdiaglogHelpMessage;

	//Constantes para modos de funcionamiento
	QString simpleImageMode;
	QString multipleImageMode;
	QString multipleAlgorythmMode;
	QString plotMode;
	QString frameCaptureMode;

	//Constantes para valores de los filtros de algoritmos
	//Sobel en x
	FloatMatrix sobelX;
	//Sobel en y.
	FloatMatrix sobelY;
	//SobelSquare en x
	FloatMatrix sobelSquareX;
	//SobelSquare en y.
	FloatMatrix sobelSquareY;
	//Prewitt en X
	FloatMatrix prewittX;
	//Prewitt en Y
	FloatMatrix prewittY;
	//RobertCross en X
	FloatMatrix robertCrossX;
	//RobertCross en Y
	FloatMatrix robertCrossY;
	//Laplace 5x5
	FloatMatrix laplace;

	//Variables de mensaje de ayuda por algoritmo.
	QString cannyHelpMessage;
	QString laplacianOfGaussianHelpMessage;
	QString laplaceHelpMessage;
	QString sobelHelpMessage;
	QString sobelSquareHelpMessage;
	QString robertCrossHelpMessage;
	QString prewittHelpMessage;

	//Métodos para inicialización de cada matriz de filtro de proceso.
	void initSobelX();
	void initSobelY();
	void initSobelSquareX();
	void initSobelSquareY();
	void initPrewittX();
	void initPrewittY();
	void initRobertCrossX();
	void initRobertCrossY();
	void initLaplace();
};

#endif /* CONSTANTS_H_ */
