/*
 * Constants.cpp
 * Propósito: clase que almacena los valores de las constantes empleadas en el programa, constantes para procesos, modos de ejecución
 *			  matrices de filtros y mensajes de ayuda de cada pantalla.
 *
 *  Creado: 21/03/2012
 *      Author: jose
 */

#include "Constants.h"

Constants* Constants::pinstance = 0;// Inicializar el puntero
Constants* Constants::Instance() {
	if (pinstance == 0) // ¿Es la primera llamada?
	{
		pinstance = new Constants; // Creamos la instancia
	}
	return pinstance; // Retornamos la dirección de la instancia
}
Constants::Constants() {
	//Definición de constantes globales para el programa.
	//Tipos de algoritmos
	this->sobelConstant = "Sobel";
	this->sobelSquaredConstant = "Sobel Square";
	this->cannyConstant = "Canny";
	this->laplaceConstant = "Laplace";
	this->laplacianOfGaussianConstant = "Laplacian Of Gaussian";
	this->prewittConstant = "Prewitt";
	this->robertCrossConstant = "Robert Cross";
	//Modos de funcionamiento
	this->simpleImageMode = "Simple Image Mode";
	this->multipleImageMode = "Multiple Image Mode";
	this->multipleAlgorythmMode = "Multiple Algorythm Mode";
	this->plotMode = "Plot Mode";
	this->frameCaptureMode = "Frame Capture Mode";

	//Mensajes de ayuda para cada panel.
	this->panel1HelpMessage
			= "<b>Ejecución simple de algoritmos</b> <br/><br/>"
			  "En esta funcionalidad de la aplicación el usuario puede ejecutar"
			  "un algoritmo sobre la imagen seleccionada, observar el resultado"
			  "del algoritmo sobre la misma y consultar los datos de ejecución de"
			  "tiempo para el mismo. "
			  "<br/><br/> - Cargue una imagen pulsado el atajo ctrl+o, el icono de la carpeta o en fichero->cargar."
			  "<br/><br/> - Seleccione un algoritmo en el desplegable y cubra los datos parámetros de ejecución en caso de ser necesarios."
			  "<br/><br/> - Ejecute el algoritmo presionando el icono de 'play', o mediante el menú fichero->ejecutar."
			  "<br/><br/> - La imagen resultado aparecerá a su derecha "
			  "<br/><br/> - Puede consultar los datos de tiempo de ejecución del algoritmo presionando el icono de información de tiempos (reloj naranja).";
	this->panel2HelpMessage = "<b>Ejecución de un algoritmo sobre múltiples imágenes</b> <br/><br/>"
			  "En esta funcionalidad de la aplicación el usuario puede ejecutar"
			  "un algoritmo sobre un conjunto de imágenes seleccionado. De este modo el usuario puede comprobar el resultado"
			  "del algoritmo sobre imágenes con diferentes características y ver sobre cual el algoritmo se comporta mejor."
			  "El usuario tendrá accesible la información de tiempo de ejecución para el algoritmo sobre las diferentes imágenes. "
			  "<br/><br/> - Cargue un conjunto de imágenes pulsado el atajo ctrl+o, el icono de la carpeta o en fichero->cargar."
			  "<br/><br/> - El conjunto de imáegnes aparecerá cargado en la mitad superior de la pantalla."
			  "<br/><br/> - Seleccione un algoritmo en el desplegable y cubra los datos parámetros de ejecución en caso de ser necesarios."
			  "<br/><br/> - Ejecute el algoritmo presionando el icono de 'play', o mediante el menú fichero->ejecutar."
			  "<br/><br/> - El conjunto de imagenes procesadas aparecerá en la mitad inferior de la pantalla. "
			  "<br/><br/> - Puede consultar los datos de tiempo de ejecución del algoritmo para cada imagen presionando el icono de información de tiempos (reloj naranja).";
	this->panel3HelpMessage = "<b>Ejecución de múltiples algoritmos</b> <br/><br/>"
			  "En esta funcionalidad de la aplicación el usuario puede ejecutar"
			  "un conjunto de algoritmos sobre la imagen seleccionada. De esta forma se puede apreciar cual es el algoritmo que mejor responde a las condiciones de la imagen"
			  "ademásde poder comparar los algoritmos deseados en cuanto al tiempo de ejecución."
			  "<br/><br/> - Cargue una imagen pulsado el atajo ctrl+o, el icono de la carpeta o en fichero->cargar."
			  "<br/><br/> - Seleccione los algoritmos deseados seleccionandolos en el desplegable, cubriendo los parámetros si es necesario y pulsando el botón añadir. "
			  "El algoritmo aparecerá en la tabla de selección mostrando sus parámetros (si un algoritmo con la misma parametrzació que la seleccionada ya se encuentra en la tabla no se podrá añadir por duplicado.)."
			  "<br/><br/> - Ejecute el el conjunto de algoritmos presionando el icono de play, o mediante el menú fichero->ejecutar."
			  "<br/><br/> - Las imágenes resultado aparecerán en la mitad inferior de la pantalla. "
			  "<be/><br/> - Puede consultar los datos de tiempo de ejecución de cada algoritmo sobre la imagen presionando el icono de información de tiempos (reloj naranja).";
	this->panel4HelpMessage = "<b>Gráficas 3D</b> <br/><br/>"
			  "En esta funcionalidad de la aplicación el usuario puede visualizar una gráfica en tres dimensiones correspondiente"
			  "al gradiente en cada punto de la imagen resultante tras aplicar el algorito seleccionado."
			  "<br/><br/> - Cargue una imagen pulsado el atajo ctrl+o, el icono de la carpeta o en fichero->cargar."
			  "<br/><br/> - Seleccione un algoritmo en el desplegable y cubra los datos parámetros de ejecución en caso de ser necesarios."
			  "<br/><br/> - Ejecute el algoritmo presionando el icono de play, o mediante el menú fichero->ejecutar."
			  "<br/><br/> - La gráfica en 3D aparecerá en la parte derecha de la pantalla. "
			  "<br/><br/> - Puede consultar los datos de tiempo de ejecución del algoritmo presionando el icono de información.";
	this->czdiaglogHelpMessage = "<b>Información sobre la tarjeta gráfica</b> <br/><br/>"
			  "En esta funcionalidad de la aplicación el usuario puede visualizar los datos de núcleo, memoria y rendimiento"
			  "de las tarjetas gráficas que tenga instaladas en su sistema.";
	this->panel6HelpMessage = "<b>Captura de frames</b> <br/><br/>"
				  "En esta funcionalidad de la aplicación el usuario puede visualizar frames capturados en tiempo real por la cámara web del sistema procesados por el algoritmo deseado.";
	this-> panel5HelpMessage= "<b>Configuración de la aplicación</b> <br/><br/>"
			  "En esta funcionalidad de la aplicación el usuario puede seleccionar la tarjeta gráfica desea para la ejecución de los algoritmos en GPU (en caso de que tenga más de una)."
			  "El usuario también podrá escoger entre tres modos de ejecución diferente. Ejecución de los algoritmos en CPU, en GPU o en ambos mostrando la comparación de tiempo."
			  "tiempo para el mismo. "
			  "<br/><br/> - Seleccione la tarjeta gráfica desea en caso de tener instalada más de una."
			  "<br/><br/> - Seleccione el modo de ejecución de la aplicación."
			  "<br/><br/> - Presione el botón guardar para aplicar los cambios realizados.";

	//Setters para las constantes de ayuda de cada algoritmo.
	this->cannyHelpMessage="<b>Algritmo de Canny</b> <br/><br/>"
			"La aplicación realizará la detección de contorno mediante la ejecución del algoritmo de canny. <br/>"
			"Este algoritmo está compuesto por cuatro etapas, son las siguientes: <br/><br/>"
			"-<b>1</b> Se difumina la imagen mediante la aplicación de un filtro gaussiano para reducir el ruido. <br/>"
			"-<b>2</b> Se calcula el valor del gradiente en cada punto de la imagen. <br/>"
			"-<b>3</b> Se realizala supresión de no-maximos en la imagen <br>"
			"     (puntos cuyo gradiente según la dirección y el valor del gradiente no sea mayor al de sus vecinos). <br/>"
			"-<b>4</b> Se realiza un proceso de histéresis para buscar las aristas reales entre dos umbrales de gradiente dados<br/>"
			"     (el umbral inferior y el superior). <br/><br/>"
			" En la ejecución de este algoritmo es posible modificar cinco parámetros, <br/>"
			"<b>radio gauss</b>, <b>sigma gauss</b>, <b>histéresis</b>, <b>umbral inferior</b> y <b>umbral superior</b>.<br/>"
			" El significado de cada uno es el siguiente: <br/><br/>"
			"- <b>radio gauss</b>: Establece el tamaño del filtro gaussiano que se aplicará para el difuminado de la imagen.<br/>"
			"- <b>sigma gauss</b>: Establece la desviación típica de los valores de la distribución gaussiana con respecto a los que se informarán los valores del filtro gaussiano para el difuminado de la imagen<br/>"
			"- <b>histéresis</b>: Permite seleccionar si se desea aplicar el paso de histéresis o no.<br/>"
			"- <b>umbral inferior</b>: Establece el valor mínimo que se comparará para considerar un punto como pertenecedor a una arista.<br/>"
			"- <b>umbral superior</b>: Establece el valor mínimo que debe tmar un punto para considerarse una arista real.<br/>";

	this->laplacianOfGaussianHelpMessage="<b>Algritmo Laplace De Gauss</b> <br/><br/>"
			"La aplicación realizará la detección de contornos mediante la ejecución del algoritmo Laplace de Gauss.<br/>"
			"Este algoritmo está compuesto por dos etapas:<br/><br/> "
			"<b>1</b>- Realización de un difuminado gaussiano sobre la imagen para la reducción del ruido<br/>"
			"<b>2</b>- Aplicación del filtro de Laplace sobre la imagen difuminada.<br/><br/>"
			"Para la ejecución de este algoritmo es posible configurar dos parámetros parámetros:<br/>"
			"<b>radio gauss</b> y <b> sigma gauss</b>. Su significado es el siguiente:<br/><br/>"
			"<b>- radio gauss: </b> Establece el tamaño del filtro gaussiano que se aplicará para el difuminado de la imagen.<br/>"
			"<b>- sigma gauss: </b> Establece la desviación típica de los valores de la distribución gaussiana<br/>"
			"con respecto a los valores que se informarán para el filtro gaussiano empleado en el difuminado de la imagen, "
			"cuanto mayor sea el valor de sigma mayor será el grado de difuminación de la imagen.";

	this->laplaceHelpMessage="<b>Algritmo de Laplace</b> <br/><br/>"
			"La aplicación realizará la detección de contornos mediante la ejecución del algoritmo de Laplace.<br/>"
			"Este algoritmo realiza una convolución de un filtro sobre la imagen, <br/>"
			"mediante la cual se obtiene la derivada de segunda orden<br/>"
			"de cada punto en la imagen, obteniendo el valor del gradiente en cada punto.<br/>"
			" Este filtro es muy sensible al ruido.";

	this->sobelHelpMessage="<b>Algritmo de Sobel</b> <br/><br/>"
			"La aplicación realizará la detección de contornos mediante la ejecución del algoritmo de Sobel.<br/>"
			"Este algoritmo realiza una convolución en X y en Y de dos filtros, calculando la aproximación a la primera derivada<br/>"
			" en cada punto de la imagen, obteniendo así el valor del gradiente en cada punto de la misma. ";

	this->sobelSquareHelpMessage="<b>Algritm Sobel Square</b> <br/><br/>"
			"La aplicación realizará la detección de contornos mediante la ejecución del algoritmo de Sobel Square.<br/>"
			"Este algoritmo realiza una convolución en X y en Y de dos filtros, calculando la aproximación a la primera derivada<br/>"
			" en cada punto de la imagen, obteniendo así el valor del gradiente en cada punto de la misma. <br/>"
			"Este algoritmo ha sido definido por C. Wang y se trata de una adaptación del algoritmo de Sobel para procesamiento<br/>"
			"de imágenes de visión nocturna o baja iluminación.";

	this->robertCrossHelpMessage="<b>Algritmo Robert Cross</b> <br/><br/>"
			"La aplicación realizará la detección de contornos mediante la ejecución del algoritmo de Robert Cross.<br/>"
			"Este algoritmo realiza una convolución de dos filtros en X e Y respectivamente consiguiendo la aproximación<br/>"
			"a la primera derivada de a imagen en cada punto. La característica principal del algoritmo de Robert Cross, es <br/>"
			"la mayor sensibilidad hacia la detección de trazos en diagonal sobre la imagen.";

	this->prewittHelpMessage="<b>Algritmo de Prewitt</b> <br/><br/>"
			"La aplicación realizará la detección de contornos mediante la ejecución del algoritmo de Prewitt.<br/>"
			"Este algoritmo realiza una convolución de filtros en X e Y sobre la imagen, consiguiente la aproximación a la<br/>"
			"primera derivada de la misma en cada punto. Este algoritmo se puede considerar una variación del algoritmo de Sobel.";

	//Matrices para los filtros
	initSobelX();
	initSobelY();
	initSobelSquareX();
	initSobelSquareY();
	initPrewittX();
	initPrewittY();
	initRobertCrossX();
	initRobertCrossY();
	initLaplace();
}

//Obtención de cada uno de los mensajes de ayuda para cada panel.
QString Constants::getPanel1HelpMessage() {
	return this->panel1HelpMessage;
}
QString Constants::getPanel2HelpMessage() {
	return this->panel2HelpMessage;
}
QString Constants::getPanel3HelpMessage() {
	return this->panel3HelpMessage;
}
QString Constants::getPanel4HelpMessage() {
	return this->panel4HelpMessage;
}
QString Constants::getPanel5HelpMessage() {
	return this->panel5HelpMessage;
}
QString Constants::getPanel6HelpMessage() {
	return this->panel6HelpMessage;
}
QString Constants::getCzdiaglogHelpMessage() {
	return this->czdiaglogHelpMessage;
}

//Obtención de las constantes correspondientes a cada modo de ejecución.
QString Constants::getSimpleImageMode() {
	return this->simpleImageMode;
}

QString Constants::getMultipleAlgorythmMode() {
	return this->multipleAlgorythmMode;
}

QString Constants::getMultipleImageMode() {
	return this->multipleImageMode;
}

QString Constants::getPlotMode() {
	return this->plotMode;
}

QString Constants::getFrameCaptureMode() {
	return this->frameCaptureMode;
}

//Obtención de las constantes para cada proceso.
QString Constants::getSobelConstant() {

	return this->sobelConstant;
}

QString Constants::getPrewittConstant() {
	return this->prewittConstant;
}

QString Constants::getCannyConstant() {

	return this->cannyConstant;
}

QString Constants::getSobelSquaredConstant() {

	return this->sobelSquaredConstant;
}

QString Constants::getLaplaceConstant() {

	return this->laplaceConstant;
}

QString Constants::getRobertCrossConstant() {

	return this->robertCrossConstant;
}

QString Constants::getLaplacianOfGaussianConstant() {

	return this->laplacianOfGaussianConstant;
}


//Métodos de obtención de cada uno de los filtros para cada proceso.
FloatMatrix Constants::getSobelX() {

	return this->sobelX;
}

FloatMatrix Constants::getSobelY() {

	return this->sobelY;
}

FloatMatrix Constants::getSobelSquareX() {

	return this->sobelSquareX;
}

FloatMatrix Constants::getSobelSquareY() {

	return this->sobelSquareY;
}

FloatMatrix Constants::getPrewittX() {

	return this->prewittX;
}

FloatMatrix Constants::getPrewittY() {

	return this->prewittY;
}

FloatMatrix Constants::getRobertCrossX() {

	return this->robertCrossX;
}

FloatMatrix Constants::getRobertCrossY() {

	return this->robertCrossY;
}

FloatMatrix Constants::getLaplace() {

	return this->laplace;
}

//Métodos de incialización de cada uno de los filtros de los procesos.
void Constants::initSobelX() {
	//    /* Sobel en x*/
	//    Gx[0] = -1.0; Gx[1] = 0.0; Gx[2] = 1.0;
	//    Gx[3] = -2.0; Gx[4] = 0.0; Gx[5] = 2.0;
	//    Gx[6] = -1.0; Gx[7] = 0.0; Gx[8] = 1.0;
	//Constantes para valores de filtros.
	this->sobelX = FloatMatrix(3, 3);
	this->sobelX.insertPos(0, 0, -1.0);
	this->sobelX.insertPos(0, 1, 0.0);
	this->sobelX.insertPos(0, 2, 1.0);
	this->sobelX.insertPos(1, 0, -2.0);
	this->sobelX.insertPos(1, 1, 0.0);
	this->sobelX.insertPos(1, 2, 2.0);
	this->sobelX.insertPos(2, 0, -1.0);
	this->sobelX.insertPos(2, 1, 0.0);
	this->sobelX.insertPos(2, 2, 1.0);

}
void Constants::initSobelY() {
	//      /*Sobel en y*/
	//      Gy[0] =  -1.0; Gy[1] =  -2.0; Gy[2] = - 1.0;
	//      Gy[3] =   0.0; Gy[4] =   0.0; Gy[5] =   0.0;
	//      Gy[6] =   1.0; Gy[7] =   2.0; Gy[8] =   1.0;
	//Constantes para valores de filtros.
	this->sobelY = FloatMatrix(3, 3);
	this->sobelY.insertPos(0, 0, -1.0);
	this->sobelY.insertPos(0, 1, -2.0);
	this->sobelY.insertPos(0, 2, -1.0);
	this->sobelY.insertPos(1, 0, 0.0);
	this->sobelY.insertPos(1, 1, 0.0);
	this->sobelY.insertPos(1, 2, 0.0);
	this->sobelY.insertPos(2, 0, 1.0);
	this->sobelY.insertPos(2, 1, 2.0);
	this->sobelY.insertPos(2, 2, 1.0);
}

void Constants::initPrewittX() {
	// Prewitt en X
	//Se establecen los filtros para prewitt.
	//	    Gx[0] = -1.0; Gx[1] =  0.0; Gx[2] =  1.0;
	//	    Gx[3] = -1.0; Gx[4] =  0.0; Gx[5] =  1.0;
	//	    Gx[6] = -1.0; Gx[7] =  0.0; Gx[8] =  1.0;
	//Constantes para valores de filtros.
	this->prewittX = FloatMatrix(3, 3);
	this->prewittX.insertPos(0, 0, -1.0);
	this->prewittX.insertPos(0, 1, 0.0);
	this->prewittX.insertPos(0, 2, 1.0);
	this->prewittX.insertPos(1, 0, -1.0);
	this->prewittX.insertPos(1, 1, 0.0);
	this->prewittX.insertPos(1, 2, 1.0);
	this->prewittX.insertPos(2, 0, -1.0);
	this->prewittX.insertPos(2, 1, 0.0);
	this->prewittX.insertPos(2, 2, 1.0);
}

void Constants::initPrewittY() {
	//      /*Prewitt en y*/
	//    Gy[0] = -1.0; Gy[1] = -1.0; Gy[2] = -1.0;
	//    Gy[3] =  0.0; Gy[4] =  0.0; Gy[5] =  0.0;
	//    Gy[6] =  1.0; Gy[7] =  1.0; Gy[8] =  1.0;
	//Constantes para valores de filtros.
	this->prewittY = FloatMatrix(3, 3);
	this->prewittY.insertPos(0, 0, -1.0);
	this->prewittY.insertPos(0, 1, -1.0);
	this->prewittY.insertPos(0, 2, -1.0);
	this->prewittY.insertPos(1, 0, 0.0);
	this->prewittY.insertPos(1, 1, 0.0);
	this->prewittY.insertPos(1, 2, 0.0);
	this->prewittY.insertPos(2, 0, 1.0);
	this->prewittY.insertPos(2, 1, 1.0);
	this->prewittY.insertPos(2, 2, 1.0);
}

void Constants::initSobelSquareX() {
	//    /* Sobel en x*/
	//    Gx[0] = -1.0; Gx[1] = 0.0; Gx[2] = 1.0;
	//    Gx[3] = -2.0; Gx[4] = 0.0; Gx[5] = 2.0;
	//    Gx[6] = -1.0; Gx[7] = 0.0; Gx[8] = 1.0;
	//Constantes para valores de filtros.
	this->sobelSquareX = FloatMatrix(3, 3);
	this->sobelSquareX.insertPos(0, 0, -1.0);
	this->sobelSquareX.insertPos(0, 1, 0.0);
	this->sobelSquareX.insertPos(0, 2, 1.0);
	this->sobelSquareX.insertPos(1, 0, -2.0);
	this->sobelSquareX.insertPos(1, 1, 0.0);
	this->sobelSquareX.insertPos(1, 2, 2.0);
	this->sobelSquareX.insertPos(2, 0, -1.0);
	this->sobelSquareX.insertPos(2, 1, 0.0);
	this->sobelSquareX.insertPos(2, 2, 1.0);

}
void Constants::initSobelSquareY() {
	//      /*Sobel en y*/
	//      Gy[0] =  -1.0; Gy[1] =  -2.0; Gy[2] = - 1.0;
	//      Gy[3] =   0.0; Gy[4] =   0.0; Gy[5] =   0.0;
	//      Gy[6] =   1.0; Gy[7] =   2.0; Gy[8] =   1.0;
	//Constantes para valores de filtros.
	this->sobelSquareY = FloatMatrix(3, 3);
	this->sobelSquareY.insertPos(0, 0, -1.0);
	this->sobelSquareY.insertPos(0, 1, -2.0);
	this->sobelSquareY.insertPos(0, 2, -1.0);
	this->sobelSquareY.insertPos(1, 0, 0.0);
	this->sobelSquareY.insertPos(1, 1, 0.0);
	this->sobelSquareY.insertPos(1, 2, 0.0);
	this->sobelSquareY.insertPos(2, 0, 1.0);
	this->sobelSquareY.insertPos(2, 1, 2.0);
	this->sobelSquareY.insertPos(2, 2, 1.0);
}

void Constants::initRobertCrossX() {
	//	 //Filtros en x e y para Robert Cross.
	//	    Gx[0] = 0.0; Gx[1] = 0.0; Gx[2] = -1.0;
	//	    Gx[3] = 0.0; Gx[4] = 1.0; Gx[5] =  0.0;
	//	    Gx[6] = 0.0; Gx[7] = 0.0; Gx[8] =  0.0;
	this->robertCrossX = FloatMatrix(3, 3);
	this->robertCrossX.insertPos(0, 0, 0.0);
	this->robertCrossX.insertPos(0, 1, 0.0);
	this->robertCrossX.insertPos(0, 2, -1.0);
	this->robertCrossX.insertPos(1, 0, 0.0);
	this->robertCrossX.insertPos(1, 1, 1.0);
	this->robertCrossX.insertPos(1, 2, 0.0);
	this->robertCrossX.insertPos(2, 0, 0.0);
	this->robertCrossX.insertPos(2, 1, 0.0);
	this->robertCrossX.insertPos(2, 2, 0.0);

}
void Constants::initRobertCrossY() {
	//	//      /*Robert cross en  y*/
	//    Gy[0] = -1.0; Gy[1] = 0.0; Gy[2] = 0.0;
	//    Gy[3] =  0.0; Gy[4] = 1.0; Gy[5] = 0.0;
	//    Gy[6] =  0.0; Gy[7] = 0.0; Gy[8] = 0.0;
	//Constantes para valores de filtros.
	this->robertCrossY = FloatMatrix(3, 3);
	this->robertCrossY.insertPos(0, 0, -1.0);
	this->robertCrossY.insertPos(0, 1, 0.0);
	this->robertCrossY.insertPos(0, 2, 0.0);
	this->robertCrossY.insertPos(1, 0, 0.0);
	this->robertCrossY.insertPos(1, 1, 1.0);
	this->robertCrossY.insertPos(1, 2, 0.0);
	this->robertCrossY.insertPos(2, 0, 0.0);
	this->robertCrossY.insertPos(2, 1, 0.0);
	this->robertCrossY.insertPos(2, 2, 0.0);
}

void Constants::initLaplace() {
	// Laplace
	//    Gy[0] = -1.0; Gy[1] = -1.0; Gy[2] = -1.0;
	//    Gy[3] = -1.0; Gy[4] =  8.0; Gy[5] = -1.0;
	//    Gy[6] = -1.0; Gy[7] = -1.0; Gy[8] = -1.0;
	//Constantes para valores de filtros.
	this->laplace = FloatMatrix(3, 3);
	this->laplace.insertPos(0, 0, -1.0);
	this->laplace.insertPos(0, 1, -1.0);
	this->laplace.insertPos(0, 2, -1.0);
	this->laplace.insertPos(1, 0, -1.0);
	this->laplace.insertPos(1, 1, 8.0);
	this->laplace.insertPos(1, 2, -1.0);
	this->laplace.insertPos(2, 0, -1.0);
	this->laplace.insertPos(2, 1, -1.0);
	this->laplace.insertPos(2, 2, -1.0);
}


//Setters para las constantes de ayuda de cada algoritmo.
QString Constants::getCannyHelpMessage(){
	return this->cannyHelpMessage;
}
QString Constants::getLaplacianOfGaussianHelpMessage(){
	return this->laplacianOfGaussianHelpMessage;
}
QString Constants::getLaplaceHelpMessage(){
	return this->laplaceHelpMessage;
}
QString Constants::getSobelHelpMessage(){
	return this->sobelHelpMessage;
}
QString Constants::getSobelSquareHelpMessage(){
	return this->sobelSquareHelpMessage;
}
QString Constants::getRobertCrossHelpMessage(){
	return this->robertCrossHelpMessage;
}
QString Constants::getPrewittHelpMessage(){
	return this->prewittHelpMessage;
}
