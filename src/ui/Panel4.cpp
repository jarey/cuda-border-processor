/*
 * Panel4.cpp : Modo Gráfica 3D
 *
 * Creado: 25/02/2012
 * Autor: jose
 */

#include "Panel4.h"
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
#include "./src/ui/graphs/include/qwt3d_io.h"
#include "./src/ui/graphs/include/qwt3d_io_gl2ps.h"
#include "./src/ui/graphs/include/qwt3d_io_reader.h"

using namespace Qwt3D;
using namespace std;

void Panel4::getDrop(const QMimeData *mimeData){
    QStringList ruta;

    if (!mimeData){}
    else{
         if(mimeData->hasUrls()) {
                  QList<QUrl> urlList = mimeData->urls();
                  QString text;
                  for (int i = 0; i < urlList.size() && i < 32; ++i) {
                      QString url = urlList.at(i).path();
                      ruta.append(url);
                  }

              } else {
              }
          }

    if(!ruta.isEmpty())
    {
    	this->loadOriginData(ruta);

    }
}

Panel4::Panel4() {
	//Se le asigna el modo de la aplicación correspondiente al panel
	Constants *constants = Constants::Instance();
	this->setApplicationMode(constants->getPlotMode());
	//Se instancian y se les da layout a los componentes
	createComponents();
	//Arrancar la interfaz gráfica con los elementos para Canny y Laplacian Of Gaussian no visibles
	showElements(constants->getCannyConstant());

	//Se setean los valores de control de habilitación de los botones de acciones en pantalla.
	setRunButtonNeeded(true);
	setStopButtonNeeded(false);
	setTimeButtonNeeded(true);
	setOpenButtonNeeded(true);
	setSaveButtonNeeded(true);

	//Se setean los manejadores de eventos para los elementos de la ventana
	connect(this->imagenOriginal, SIGNAL(changed(const QMimeData*)),this,SLOT(getDrop(const QMimeData*)));
	connect(this->algorythmSelect, SIGNAL(activated(int)),this, SLOT(consultarCombo()));
	connect(this->botonPrueba,SIGNAL(clicked()),this,SLOT(showAlgorythmHelpMessage()));
}

Panel4::~Panel4(){
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

}


void Panel4::createComponents() {
	//Se instancian y configuran los componentes
	this->instantiateElements();
	//Se les da el layout deseado dentro del panel
	this->layoutComponents();
}

void Panel4::instantiateElements() {

	packagesGroup = new QGroupBox(tr("Gráfica De Gradiente"));
	//Layout horizontal
	horizontalLayout1 = new QHBoxLayout();
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

	imagenOriginalLabel = new QLabel("Imagen Original");
	imagenOriginal = new DropArea();
	//Segunda columna
	imagenResultadoLabel = new QLabel("Gráfica Resultado");
	imagenResultado = new PlotWidget();

	//Label algoritmos
	algoritmosLabel = new QLabel("Algoritmo: ");
	algoritmosLabel->setContentsMargins(0,0,0,0);
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
	sigmaGauss  = new QDoubleSpinBox();
	sigmaGauss->setValue(1);
	sigmaGauss->setSingleStep(0.1);
	sigmaGauss->setMinimum(1);
	sigmaGauss->setMaximum(10);
	umbralSuperior  = new QDoubleSpinBox();
	umbralSuperior->setMaximumSize(QSize(100, 16777215));
	umbralSuperior->setDecimals(3);
	umbralSuperior->setMinimum(0);
	umbralSuperior->setMaximum(85);
	umbralSuperior->setSingleStep(0.001);
	umbralSuperior->setValue(0.200);
	umbralInferior  = new QDoubleSpinBox();
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

	horizontalSpacer = new QSpacerItem(20, 10, QSizePolicy::Maximum, QSizePolicy::Maximum);
	horizontalSpacer_2 = new QSpacerItem(2000, 10, QSizePolicy::Maximum, QSizePolicy::Maximum);

	botonPrueba = new QPushButton();
	botonPrueba->setStyleSheet("background:transparent;border-radius:5px;;max-width:20px;max-height:20px;min-width:20px;min-height:20px;");
	botonPrueba->setIcon(QIcon(":images/help.png"));
}

void Panel4::layoutComponents() {
	//Posicionamiento de los elementos en los layouts
	//Columnas verticales para label/lienzo
	verticalLayout1->addWidget(imagenOriginalLabel);
	verticalLayout1->setSpacing(1);
	verticalLayout1->addWidget(imagenOriginal);
	horizontalLayout1->addLayout(verticalLayout1);
	imagenResultadoLabel->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum);
	imagenResultadoLabel->setFixedHeight(12);
	verticalLayout2->addWidget(imagenResultadoLabel);
	verticalLayout2->setSpacing(1);
	imagenResultado->setMinimumHeight(imagenOriginal->height());
	imagenResultado->setMinimumWidth(imagenOriginal->width());
	verticalLayout2->addWidget(imagenResultado);

	horizontalLayout1->addLayout(verticalLayout2);

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



void Panel4::consultarCombo()
{
	Controlador *controlador = Controlador::Instance();
	controlador->setAlgorythmSelected(algorythmSelect->currentText());
    showElements(algorythmSelect->currentText());
}

void Panel4::showElements(QString displayMode)
{
	Constants *contants = Constants::Instance();

    if(displayMode == contants->getCannyConstant()){
    	//Se muestran los elementos adicionales para algoritmo canny
		histeresisCheck->setVisible(true);
		histeresisLabel->setVisible(true);
		umbralInferior->setVisible(true);
		umbralSuperior->setVisible(true);
		umbralInferiorLabel->setVisible(true);
		umbralSuperiorLabel->setVisible(true);;
		radioGauss->setVisible(true);
		radioGaussLabel->setVisible(true);
		sigmaGauss->setVisible(true);
		sigmaGaussLabel->setVisible(true);
    }else if(displayMode == contants->getLaplacianOfGaussianConstant()){
    	histeresisCheck->setVisible(false);
    	histeresisLabel->setVisible(false);
    	umbralInferior->setVisible(false);
    	umbralSuperior->setVisible(false);
    	umbralInferiorLabel->setVisible(false);
    	umbralSuperiorLabel->setVisible(false);;
    	radioGauss->setVisible(true);
    	radioGaussLabel->setVisible(true);
    	sigmaGauss->setVisible(true);
    	sigmaGaussLabel->setVisible(true);
    }else{
    	histeresisCheck->setVisible(false);
    	histeresisLabel->setVisible(false);
    	umbralInferior->setVisible(false);
    	umbralSuperior->setVisible(false);
    	umbralInferiorLabel->setVisible(false);
    	umbralSuperiorLabel->setVisible(false);;
    	radioGauss->setVisible(false);
    	radioGaussLabel->setVisible(false);
    	sigmaGauss->setVisible(false);
    	sigmaGaussLabel->setVisible(false);
    }
}

void Panel4::loadOriginalImage(QString ruta){
    if(!ruta.isNull())
    {
    	Controlador *controlador = Controlador::Instance();

        QPixmap imagen =QPixmap(ruta);
        //Se setea la imagen cargada en el controlador.
        controlador->setOriginImage(imagen.toImage());
        QGraphicsScene *scn = new QGraphicsScene( this->imagenOriginal );
        scn->addPixmap(imagen);
        this->imagenOriginal->setScene(scn);
        /*Rellenado de la imagen "clon" de blanco*/
        QPixmap imagen2=QPixmap(imagen.width(),imagen.height());
        imagen2.fill(Qt::white);
        QImage imagen3=imagen.toImage();
        imagenResultado->setMinimumHeight(imagenOriginal->height());
        imagenResultado->setMinimumWidth(imagenOriginal->width());
    }
}
void Panel4::loadResultImage(QImage resultImage){

}

void Panel4::loadOriginData(QStringList list){
	QString ruta = list.at(0);
	if(!ruta.isNull())
	    {
		this->getMatrixListOrigin()->clear();
	    	Controlador *controlador = Controlador::Instance();

	        QPixmap imagen =QPixmap(ruta);
	        //Se setea la imagen cargada en el controlador.
	        controlador->setOriginImage(imagen.toImage());
	        controlador->getMatrixListOrigin()->append(imagen.toImage());
	        QGraphicsScene *scn = new QGraphicsScene( this->imagenOriginal );
	        scn->addPixmap(imagen);
	        this->imagenOriginal->setScene(scn);
	        /*Rellenado de la imagen "clon" de blanco*/
	        delete imagenResultado;
	    	imagenResultado = new PlotWidget();
	    	verticalLayout2->addWidget(imagenResultado);
	        imagenResultado->setMinimumHeight(imagenOriginal->height());
	        imagenResultado->setMinimumWidth(imagenOriginal->width());
	        this->getMatrixListOrigin()->append(imagen.toImage());
	    }

}

QList<QImage> Panel4::getOriginData(){

}

void Panel4::loadResultData(QList<QImage> *imageList){

	if(!imageList->isEmpty()){
		this->getMatrixListResult()->clear();
		QPixmap imagen = QPixmap();
		imagen = imagen.fromImage(imageList->at(0));
		QGraphicsScene *scn = new QGraphicsScene( this->imagenOriginal );
		scn->addPixmap(imagen);
		this->imagenOriginal->setScene(scn);
		//Se pinta la gráfica para el panel derecho.
		this->callPlot(imageList->at(0));
		this->getMatrixListResult()->append(imageList->at(0));
	}else{
		QMessageBox::warning(this, tr("Error!"), tr("Error en el proceso de carga de la imagen resultado."));
	}
}


QList<QImage> Panel4::getResultData(){

}

void Panel4::callPlot(QImage image){
	//Se llama a la función del pintado de la gráfica
	this->imagenResultado->doPlot(image);
}

QString Panel4::getHelpData(){
	return Constants::Instance()->getPanel4HelpMessage();
}

QString Panel4::getExecutionData(){

}

void Panel4::saveImages(){

		if(!this->getMatrixListResult()->isEmpty()){

			QString fileName = QFileDialog::getSaveFileName(this, tr("Guardar gráfica 3D como imagen"),
											QDir::homePath(),
			                            tr("Imagenes (*.jpg)"));

			if ( fileName.isNull() == false )
			{
				IO::save(this->imagenResultado->plot,fileName+".jpg","jpg");
			}

		}else{
		}
}

//Obtiene el String de información correspondiente para el panel
QString Panel4::getAlgorythmHelpData(){
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

void Panel4::showAlgorythmHelpMessage() {

	QMessageBox::about(this, tr("Ayuda"), this->getAlgorythmHelpData());

}
