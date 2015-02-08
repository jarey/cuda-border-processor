/*
 * Panel2.cpp : Panel de Ejecución MULTIPLES IMÁGENES
 *
 * Creado: 25/02/2012
 * Autor: jose
 */

#include "Panel2.h"
#include <QtGui>
#include <QFileDialog>
#include <stdio.h>
#include <QImage>
#include <QGraphicsEllipseItem>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QPixmap>
#include <math.h>
#include <queue>
#include <QtDebug>
#include <qdialog.h>
#include "./src/common/Constants.h"
#include "./src/common/Controlador.h"
#include <QSizePolicy>

void Panel2::getDrop(const QMimeData *mimeData) {
	QStringList ruta;

	if (!mimeData) {
	} else {
		if (mimeData->hasUrls()) {
			QList<QUrl> urlList = mimeData->urls();
			QString text;
			for (int i = 0; i < urlList.size() && i < 32; ++i) {
				QString url = urlList.at(i).path();
				ruta.append(url);
			}
		} else {
		}
	}
	if (!ruta.isEmpty()) {
		this->loadOriginData(ruta);
	}
}

Panel2::Panel2() {
	//Se le asigna el modo de la aplicación correspondiente al panel
	Constants *constants = Constants::Instance();
	this->setApplicationMode(constants->getMultipleImageMode());
	//Se instancian y se les da layout a los componentes
	createComponents();
	//Arrancar la interfaz gráfica con los elementos para Canny y Laplacian Of Gaussian no visibles
	showElements(Constants::Instance()->getCannyConstant());

	//Se setean los valores de control de habilitación de los botones de acciones en pantalla.
	setRunButtonNeeded(true);
	setStopButtonNeeded(false);
	setTimeButtonNeeded(true);
	setOpenButtonNeeded(true);
	setSaveButtonNeeded(true);

	//Se setean los manejadores de eventos para los elementos de la ventana
	connect(this->imagenOriginal, SIGNAL(changed(const QMimeData*)), this,
			SLOT(getDrop(const QMimeData*)));
	connect(this->algorythmSelect, SIGNAL(activated(int)), this,
			SLOT(consultarCombo()));
	connect(this->botonPrueba,SIGNAL(clicked()),this,SLOT(showAlgorythmHelpMessage()));

}

Panel2::~Panel2(){
	delete packagesGroup;
	//Layout horizontal
	delete horizontalLayout1;
	//Layouts verticales para formar las 2 columnas
	delete verticalLayout1;
	delete verticalLayout2;
	delete verticalLayout3;
	//Layout por grupos label/componente
	delete labelComboAlgoritmo;
	delete labelCheckHisteresis;
	delete labelSpinnerRadioGaus;
	delete labelSpinnerSigmaGaus;
	delete labelSpinnerUmbralSuperior;
	delete labelSpinnerUmbralInferior;
	//Primera columna
	//QSizePolicy sizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	delete imagenOriginalLabel;
	delete imagenOriginal;
	//Segunda columna
	delete imagenResultadoLabel;
	delete imagenResultado;
	//Label algoritmos
	delete algoritmosLabel;
	//Horizontal layout para el label y el combo
	delete horizontalLayout2;
	delete algorythmSelect;
	delete histeresisCheck;
	delete radioGauss;
	delete sigmaGauss;
	delete umbralSuperior;
	delete umbralInferior;
	//Elementos del combo
	delete algorythmSelect;
	//Histeresis
	delete histeresisLabel;
	//Sigma gauss
	delete sigmaGaussLabel;
	//Radio gauss
	delete radioGaussLabel;
	//Umbral superior
	delete umbralSuperiorLabel;
	//Umbral inferior
	delete umbralInferiorLabel;

	delete horizontalSpacer;
	delete horizontalSpacer_2;
	delete scrollArea;
	delete scrollArea2;

}

void Panel2::createComponents() {
	//Se instancian y configuran los componentes
	this->instantiateElements();
	//Se les da el layout deseado dentro del panel
	this->layoutComponents();
}

void Panel2::instantiateElements() {

	packagesGroup = new QGroupBox(
			tr("Ejecución de algoritmo sobre múltiples imágenes."));
	//Layout horizontal
	horizontalLayout1 = new QVBoxLayout();
	//Layouts verticales para formar las 2 columnas
	verticalLayout1 = new QVBoxLayout();
	verticalLayout2 = new QVBoxLayout();
	verticalLayout3 = new QVBoxLayout();
	//Layout por grupos label/componente
	labelComboAlgoritmo = new QHBoxLayout();
	labelCheckHisteresis = new QHBoxLayout();
	labelSpinnerRadioGaus = new QHBoxLayout();
	labelSpinnerSigmaGaus = new QHBoxLayout();
	labelSpinnerUmbralSuperior = new QHBoxLayout();
	labelSpinnerUmbralInferior = new QHBoxLayout();
	//Primera columna
	//QSizePolicy sizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	imagenOriginalLabel = new QLabel("Imagen Original");
	imagenOriginal = new DropArea();
	//Segunda columna
	imagenResultadoLabel = new QLabel("Imagen Procesada");
	imagenResultado = new QGraphicsView();
	//Label algoritmos
	algoritmosLabel = new QLabel("Algoritmo: ");
	algoritmosLabel->setContentsMargins(0, 0, 0, 0);
	//Horizontal layout para el label y el combo
	horizontalLayout2 = new QHBoxLayout();
	algorythmSelect = new QComboBox();
	algorythmSelect->setMaximumWidth(200);
	Constants *p2 = Constants::Instance();
	histeresisCheck = new QCheckBox();
	histeresisCheck->setChecked(true);
	radioGauss = new QSpinBox();
	radioGauss->setValue(1);
	radioGauss->setSingleStep(2);
	radioGauss->setMinimum(1);
	radioGauss->setMaximum(25);
	sigmaGauss = new QDoubleSpinBox();
	sigmaGauss->setValue(1);
	sigmaGauss->setSingleStep(0.1);
	sigmaGauss->setMinimum(1);
	sigmaGauss->setMaximum(10);
	umbralSuperior = new QDoubleSpinBox();
	umbralSuperior->setMaximumSize(QSize(100, 16777215));
	umbralSuperior->setDecimals(3);
	umbralSuperior->setMinimum(0);
	umbralSuperior->setMaximum(85);
	umbralSuperior->setSingleStep(0.001);
	umbralSuperior->setValue(0.200);
	umbralInferior = new QDoubleSpinBox();
	umbralInferior->setMaximumSize(QSize(100, 16777215));
	umbralInferior->setDecimals(3);
	umbralInferior->setMinimum(0);
	umbralInferior->setMaximum(85);
	umbralInferior->setSingleStep(0.001);
	umbralInferior->setValue(0.020);
	//Elementos del combo
	algorythmSelect->addItem(p2->getCannyConstant());
	algorythmSelect->addItem(p2->getLaplacianOfGaussianConstant());
	algorythmSelect->addItem(p2->getSobelConstant());
	algorythmSelect->addItem(p2->getSobelSquaredConstant());
	algorythmSelect->addItem(p2->getPrewittConstant());
	algorythmSelect->addItem(p2->getRobertCrossConstant());
	algorythmSelect->addItem(p2->getLaplaceConstant());


	//Histeresis
	histeresisLabel = new QLabel("Histeresis: ");
	//Sigma gauss
	sigmaGaussLabel = new QLabel("Sigma Gauss: ");
	//Radio gauss
	radioGaussLabel = new QLabel("Radio Gauss: ");
	//Umbral superior
	umbralSuperiorLabel = new QLabel("Umbral superior: ");
	//Umbral inferior
	umbralInferiorLabel = new QLabel("Umbral inferior: ");

	horizontalSpacer = new QSpacerItem(20, 10, QSizePolicy::Maximum,
			QSizePolicy::Maximum);
	horizontalSpacer_2 = new QSpacerItem(2000, 10, QSizePolicy::Maximum,
			QSizePolicy::Maximum);
	scrollArea = new QScrollArea();
	scrollArea2 = new QScrollArea();

	botonPrueba = new QPushButton();
	botonPrueba->setStyleSheet("background:transparent;border-radius:5px;;max-width:20px;max-height:20px;min-width:20px;min-height:20px;");
	botonPrueba->setIcon(QIcon(":images/help.png"));
}

void Panel2::layoutComponents() {
	//Posicionamiento de los elementos en los layouts
	//Columnas verticales para label/lienzo
	verticalLayout1->addWidget(imagenOriginalLabel);
	verticalLayout1->setSpacing(1);
	verticalLayout1->addWidget(imagenOriginal);
	horizontalLayout1->addWidget(scrollArea);
	horizontalLayout1->addWidget(scrollArea2);
	verticalLayout2->addWidget(imagenResultadoLabel);
	verticalLayout2->setSpacing(1);
	verticalLayout2->addWidget(imagenResultado);

	//Grupos label/componente para cabecera horizotal
	labelComboAlgoritmo->addWidget(algoritmosLabel);
	labelComboAlgoritmo->addWidget(algorythmSelect);
	labelComboAlgoritmo->addWidget(botonPrueba);
	horizontalLayout2->addLayout(labelComboAlgoritmo);
	horizontalLayout2->addItem(horizontalSpacer);
	labelCheckHisteresis->addWidget(histeresisLabel);
	labelCheckHisteresis->addWidget(histeresisCheck);
	horizontalLayout2->addLayout(labelCheckHisteresis);
	labelSpinnerRadioGaus->addWidget(radioGaussLabel);
	labelSpinnerRadioGaus->addWidget(radioGauss);
	horizontalLayout2->addLayout(labelSpinnerRadioGaus);
	labelSpinnerSigmaGaus->addWidget(sigmaGaussLabel);
	labelSpinnerSigmaGaus->addWidget(sigmaGauss);
	horizontalLayout2->addLayout(labelSpinnerSigmaGaus);
	labelSpinnerUmbralInferior->addWidget(umbralInferiorLabel);
	labelSpinnerUmbralInferior->addWidget(umbralInferior);
	horizontalLayout2->addLayout(labelSpinnerUmbralInferior);
	labelSpinnerUmbralSuperior->addWidget(umbralSuperiorLabel);
	labelSpinnerUmbralSuperior->addWidget(umbralSuperior);
	horizontalLayout2->addLayout(labelSpinnerUmbralSuperior);
	horizontalLayout2->addItem(horizontalSpacer_2);

	//Posteriormente el layout de cabecera y el de las columnas al layout final
	verticalLayout3->addLayout(horizontalLayout2);
	verticalLayout3->addLayout(horizontalLayout1);
	verticalLayout3->setAlignment(Qt::AlignLeft);
	//Seteo del layout final grupo
	packagesGroup->setLayout(verticalLayout3);
	//Creo un layout más para añadir el grupo a este layout
	QVBoxLayout *verticalLayoutAux = new QVBoxLayout();
	verticalLayoutAux->addWidget(packagesGroup);
	setLayout(verticalLayoutAux);
}

void Panel2::consultarCombo() {
	Controlador *controlador = Controlador::Instance();
	controlador->setAlgorythmSelected(algorythmSelect->currentText());
	showElements(algorythmSelect->currentText());
}

void Panel2::showElements(QString displayMode) {
	Constants *contants = Constants::Instance();

	if (displayMode == contants->getCannyConstant()) {
		//Se muestran los elementos adicionales para algoritmo canny
		histeresisCheck->setVisible(true);
		histeresisLabel->setVisible(true);
		umbralInferior->setVisible(true);
		umbralSuperior->setVisible(true);
		umbralInferiorLabel->setVisible(true);
		umbralSuperiorLabel->setVisible(true);
		radioGauss->setVisible(true);
		radioGaussLabel->setVisible(true);
		sigmaGauss->setVisible(true);
		sigmaGaussLabel->setVisible(true);
	} else if (displayMode == contants->getLaplacianOfGaussianConstant()) {
		histeresisCheck->setVisible(false);
		histeresisLabel->setVisible(false);
		umbralInferior->setVisible(false);
		umbralSuperior->setVisible(false);
		umbralInferiorLabel->setVisible(false);
		umbralSuperiorLabel->setVisible(false);
		radioGauss->setVisible(true);
		radioGaussLabel->setVisible(true);
		sigmaGauss->setVisible(true);
		sigmaGaussLabel->setVisible(true);
	} else {
		histeresisCheck->setVisible(false);
		histeresisLabel->setVisible(false);
		umbralInferior->setVisible(false);
		umbralSuperior->setVisible(false);
		umbralInferiorLabel->setVisible(false);
		umbralSuperiorLabel->setVisible(false);
		radioGauss->setVisible(false);
		radioGaussLabel->setVisible(false);
		sigmaGauss->setVisible(false);
		sigmaGaussLabel->setVisible(false);
	}
}

void Panel2::loadResultImage(QImage resultImage) {
	QPixmap imagen = QPixmap();
	imagen = imagen.fromImage(resultImage);
	QGraphicsScene *scn = new QGraphicsScene(this->imagenResultado);
	scn->addPixmap(imagen);
	this->imagenResultado->setScene(scn);
}

void Panel2::loadOriginData(QStringList list) {

	if (!list.isEmpty()) {
		this->getMatrixListOrigin()->clear();

		Controlador *controlador = Controlador::Instance();

		QHBoxLayout *horizontalLayout = new QHBoxLayout();
		QWidget *widget = new QWidget();
		for (int i = 0; i < list.size(); ++i) {
			QHBoxLayout *horizontalLayout1 = new QHBoxLayout();
			QGraphicsView *qImage = new QGraphicsView();
			qImage->setMaximumHeight(scrollArea->height() - 10);
			qImage->setMaximumWidth(scrollArea->height() - 10);

			QPixmap imagen = QPixmap(list.at(i));
			QPixmap scaledImage;
			if (imagen.height() > scrollArea->height() - 10 || imagen.width()
					> scrollArea->height() - 10) {
				scaledImage = imagen.scaledToHeight(scrollArea->height() - 10,
						Qt::FastTransformation);

				this->getMatrixListOrigin()->append(imagen.toImage());

				imagen = scaledImage.scaledToWidth(scrollArea->height() - 10,
						Qt::FastTransformation);
			}
			QGraphicsScene *scn = new QGraphicsScene(this->imagenOriginal);
			scn->addPixmap(imagen);

			qImage->setScene(scn);
			qreal scaleFactorx = (float) scrollArea->height() / imagen.width()
					- ((float) scrollArea->height() / imagen.width());
			qreal scaleFactory = (float) scrollArea->height() / imagen.width()
					- ((float) scrollArea->height() / imagen.width());
			horizontalLayout1->addWidget(qImage);
			horizontalLayout->addLayout(horizontalLayout1);
		}
		widget->setLayout(horizontalLayout);
		this->scrollArea->setWidget(widget);
		QWidget *wi = new QWidget();
		this->scrollArea2->setWidget(wi);
	}
}

QList<QImage> Panel2::getOriginData() {

}

void Panel2::loadResultData(QList<QImage> *imageList) {

	if (!imageList->isEmpty()) {
		this->getMatrixListResult()->clear();
		Controlador *controlador = Controlador::Instance();
		QHBoxLayout *horizontalLayout = new QHBoxLayout();
		QWidget *widget = new QWidget();
		for (int i = 0; i < imageList->size(); ++i) {
			QHBoxLayout *horizontalLayout1 = new QHBoxLayout();
			QGraphicsView *qImage = new QGraphicsView();
			qImage->setMaximumHeight(scrollArea->height() - 10);
			qImage->setMaximumWidth(scrollArea->height() - 10);

			QPixmap imagen = QPixmap();
			imagen = imagen.fromImage(imageList->at(i));
			this->getMatrixListResult()->append(imageList->at(i));
			QPixmap scaledImage;
			if (imagen.height() > scrollArea2->height() - 10 || imagen.width()
					> scrollArea2->height() - 10) {
				scaledImage = imagen.scaledToHeight(scrollArea->height() - 10,
						Qt::FastTransformation);

				imagen = scaledImage.scaledToWidth(scrollArea->height() - 10,
						Qt::FastTransformation);
			}
			QGraphicsScene *scn = new QGraphicsScene(this->imagenOriginal);
			scn->addPixmap(imagen);

			qImage->setScene(scn);
			qreal scaleFactorx = (float) scrollArea2->height() / imagen.width()
					- ((float) scrollArea2->height() / imagen.width());
			qreal scaleFactory = (float) scrollArea2->height() / imagen.width()
					- ((float) scrollArea2->height() / imagen.width());
			horizontalLayout1->addWidget(qImage);
			horizontalLayout->addLayout(horizontalLayout1);
		}
		widget->setLayout(horizontalLayout);
		this->scrollArea2->setWidget(widget);
	}

}

QList<QImage> Panel2::getResultData() {

}

QString Panel2::getHelpData(){
	return Constants::Instance()->getPanel2HelpMessage();
}



QString Panel2::getExecutionData(){
	QString executionData;
	QString val;
	for(int i=0; i<Controlador::Instance()->getMultipleImageExecution()->size();i++){
		TimeContainer timeContainer;
		timeContainer = Controlador::Instance()->getMultipleImageExecution()->at(i);
		//Mensaje para ejecución simple del algoritmo bien sea CPU o GPU
		if(timeContainer.getExecutionType() == "CPU"){
			executionData += "<b>Algoritmo</b> : "+ timeContainer.getProcess() + "<br/>";
			executionData += "Tipo de ejecución : " + timeContainer.getExecutionType() + "<br/>";
			executionData += "Tiempo de ejecución : " +  val.setNum(timeContainer.getCPUMilliseconds()) + " milisegundos <br/>";
			executionData += "<br/><br/>";
		}else if(timeContainer.getExecutionType() == "GPU"){
			executionData += "<b>Algoritmo</b> : "+ timeContainer.getProcess() + "<br/>";
			executionData += "Tipo de ejecución : " + timeContainer.getExecutionType() + "<br/>";
			executionData += "Tiempo de ejecución : " +  val.setNum(timeContainer.getGPUMilliseconds()) + " milisegundos <br/>";
			executionData += "<br/><br/>";
		}else if(timeContainer.getExecutionType() == "CPU vs GPU"){//Caso para CPU vs GPU
			executionData += "<b>Algoritmo</b> : "+ timeContainer.getProcess() + "<br/>";
			executionData += "Tipo de ejecución : " + timeContainer.getExecutionType() + "<br/>";
			executionData += "Tiempo de ejecución CPU: " +  val.setNum(timeContainer.getCPUMilliseconds()) + " milisegundos <br/>";
			executionData += "Tiempo de ejecución GPU: " +  val.setNum(timeContainer.getGPUMilliseconds()) + " milisegundos <br/>";
			executionData += "Aceleración conseguida en GPU con respecto a CPU: " +  val.setNum(timeContainer.getGraphiAceleration()) + " milisegundos <br/>";
			//Aceleración falta
			executionData += "<br/><br/>";
		}
	}

	return executionData;
}

void Panel2::saveImages(){
	QString val;
	if(!this->getMatrixListResult()->isEmpty()){
		for(int i=0; i<this->getMatrixListResult()->size();i++){
		QString message;
		int numberTotal = this->getMatrixListResult()->size();
		int numberActual = i+1;
		message += "Guardar imagen "+ QString::number(numberActual) + " de "+ QString::number(numberTotal) + "";
		QString fileName = QFileDialog::getSaveFileName(this, message,
		                            QDir::homePath(),
		                            tr("Images (*.jpg)"));

		if ( fileName.isNull() == false )
		{
			this->getMatrixListResult()->at(i).save(fileName+".jpg","JPG");
		}

		}

	}else{
	}

}

//Obtiene el String de información correspondiente para el panel
QString Panel2::getAlgorythmHelpData(){
	QString message;
		Constants *constantes = Constants::Instance();
		//Realiza la obtención del mensaje de ayuda del algoritmo en función del seleccionado en el combo del panel.
		//Canny
		if(this->algorythmSelect->currentText()== constantes->getCannyConstant()){
			message = constantes->getCannyHelpMessage();
		//Laplace de Gauss
		}else if(this->algorythmSelect->currentText() == constantes->getLaplacianOfGaussianConstant()){
			message = constantes->getLaplacianOfGaussianHelpMessage();
		//Laplace
		}else if(this->algorythmSelect->currentText() ==constantes->getLaplaceConstant()){
			message = constantes->getLaplaceHelpMessage();
		//Sobel
		}else if(this->algorythmSelect->currentText() ==constantes->getSobelConstant()){
			message = constantes->getSobelHelpMessage();
		//Sobel Square
		}else if(this->algorythmSelect->currentText() ==constantes->getSobelSquaredConstant()){
			message = constantes->getSobelSquareHelpMessage();
		//Prewitt
		}else if(this->algorythmSelect->currentText() ==constantes->getPrewittConstant()){
			message = constantes->getPrewittHelpMessage();
		//Robert Cross
		}else if(this->algorythmSelect->currentText() ==constantes->getRobertCrossConstant()){
			message = constantes->getRobertCrossHelpMessage();
		}
		return message;
}

void Panel2::showAlgorythmHelpMessage() {

	QMessageBox::about(this, tr("Ayuda"), this->getAlgorythmHelpData());

}
