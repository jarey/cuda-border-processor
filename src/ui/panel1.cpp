/*
 * panel1.cpp : Panel de EJECUCIÓN SIMPLE
 *
 * Creado: 25/02/2012
 * Autor: jose
 */

#include "panel1.h"
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
#include "./src/common/TimeContainer.h"
//***


void panel1::getDrop(const QMimeData *mimeData){
    QStringList ruta;

    if (!mimeData){}
    else{
        printf("\n 1 \n");
         if(mimeData->hasUrls()) {
             printf("\n 2 \n");
                  QList<QUrl> urlList = mimeData->urls();
                  printf("\n 3 \n");
                  QString text;
                  printf("\n 4 \n");
                  for (int i = 0; i < urlList.size() && i < 32; ++i) {
                      printf("\n 5 \n");
                      QString url = urlList.at(i).path();
                      ruta.append(url);
                  }
              } else {
                    printf("\n 9 \n");
              }
          }
    printf("\n 10 \n");
    if(!ruta.isEmpty())
    {

	this->loadOriginData(ruta);

    }
}

panel1::panel1() {
	//Se le asigna el modo de la aplicación correspondiente al panel
	Constants *constants = Constants::Instance();
	this->setApplicationMode(constants->getSimpleImageMode());

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

panel1::~panel1(){
	//Elemntos simples
	delete algorythmSelect;
	delete histeresisCheck;
	delete radioGauss;
	delete sigmaGauss;
	delete umbralSuperior;
	delete umbralInferior;
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
	//Label algoritmos
	delete algoritmosLabel;
	delete imagenOriginalLabel;
	delete imagenOriginal;
	//Segunda columna
	delete imagenResultadoLabel;
	delete imagenResultado;
	delete labelComboAlgoritmo;
	delete labelCheckHisteresis;
	delete labelSpinnerRadioGaus;
	delete labelSpinnerSigmaGaus;
	delete labelSpinnerUmbralSuperior;
	delete labelSpinnerUmbralInferior;

	delete horizontalSpacer;
	delete horizontalSpacer_2;
	delete verticalLayout2;
	delete verticalLayout1;
	delete horizontalLayout1;
	//delete horizontalLayout2;
	//delete verticalLayout3;
	//delete packagesGroup;
}


void panel1::createComponents() {
	//Se instancian y configuran los componentes
	this->instantiateElements();
	//Se les da el layout deseado dentro del panel
	this->layoutComponents();
}

void panel1::instantiateElements() {

	packagesGroup = new QGroupBox(tr("Ejecución simple"));
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
	//QSizePolicy sizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	imagenOriginalLabel = new QLabel("Imagen Original");
	imagenOriginal = new DropArea();
	//Segunda columna
	imagenResultadoLabel = new QLabel("Imagen Procesada");
	imagenResultado = new QGraphicsView();
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

void panel1::layoutComponents() {
	//Posicionamiento de los elementos en los layouts
	//Columnas verticales para label/lienzo
	verticalLayout1->addWidget(imagenOriginalLabel);
	verticalLayout1->setSpacing(1);
	verticalLayout1->addWidget(imagenOriginal);
	horizontalLayout1->addLayout(verticalLayout1);
	verticalLayout2->addWidget(imagenResultadoLabel);
	verticalLayout2->setSpacing(1);
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



void panel1::consultarCombo()
{
	Controlador *controlador = Controlador::Instance();
	controlador->setAlgorythmSelected(algorythmSelect->currentText());
    showElements(algorythmSelect->currentText());
}

void panel1::showElements(QString displayMode)
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

void panel1::loadOriginalImage(QString ruta){
    if(!ruta.isNull())
    {
    	Controlador *controlador = Controlador::Instance();

        QPixmap imagen =QPixmap(ruta);
        //Se setea la imagen cargada en el controlador.
        //controlador->setOriginImage(imagen.toImage());
        this->getMatrixListOrigin()->clear();
        this->getMatrixListOrigin()->append(imagen.toImage());
        QGraphicsScene *scn = new QGraphicsScene( this->imagenOriginal );
        scn->addPixmap(imagen);
        this->imagenOriginal->setScene(scn);
        /*Rellenado de la imagen "clon" de blanco*/
        QPixmap imagen2=QPixmap(imagen.width(),imagen.height());
        imagen2.fill(Qt::white);
        QImage imagen3=imagen.toImage();
        QGraphicsScene *scn2 = new QGraphicsScene( this->imagenResultado );
        scn2->addPixmap(imagen2);
        this->imagenResultado->setScene(scn2);


    }
}
void panel1::loadResultImage(QImage resultImage){

		QPixmap imagen = QPixmap();
		imagen = imagen.fromImage(resultImage);
		QGraphicsScene *scn = new QGraphicsScene( this->imagenResultado );
		scn->addPixmap(imagen);
		this->imagenResultado->setScene(scn);
	}

void panel1::loadOriginData(QStringList list){
	QString ruta = list.at(0);
	if(!ruta.isNull())
	    {
		qDebug() << "Entro aqui en el panel 1 carga de imagen." ;
	    	Controlador *controlador = Controlador::Instance();

	        QPixmap imagen =QPixmap(ruta);
	        //Se setea la imagen cargada en el controlador.
	        qDebug() << "El numero de imágenes en el origen ANTES es: " << this->getMatrixListOrigin()->size();
	        this->getMatrixListOrigin()->clear();
	        this->getMatrixListOrigin()->append(imagen.toImage());
	        this->getMatrixListResult()->clear();
	        qDebug() << "El numero de imágenes en el origen DESPEUS es: " << this->getMatrixListOrigin()->size();
	        QGraphicsScene *scn = new QGraphicsScene( this->imagenOriginal );
	        scn->addPixmap(imagen);
	        this->imagenOriginal->setScene(scn);
	        /*Rellenado de la imagen "clon" de blanco*/
	        QPixmap imagen2=QPixmap(imagen.width(),imagen.height());
	        imagen2.fill(Qt::white);
	        QImage imagen3=imagen.toImage();
	        QGraphicsScene *scn2 = new QGraphicsScene( this->imagenResultado );
	        scn2->addPixmap(imagen2);
	        this->imagenResultado->setScene(scn2);
	        qDebug() << "Salgo de  aqui en el panel 1 carga de imagen." ;
	    }

}

QList<QImage> panel1::getOriginData(){

}

void panel1::loadResultData(QList<QImage> *imageList){

	if(!imageList->isEmpty()){
		QPixmap imagen = QPixmap();
		imagen = imagen.fromImage(imageList->at(0));
		this->getMatrixListResult()->clear();
		this->getMatrixListResult()->append(imageList->at(0));
		QGraphicsScene *scn = new QGraphicsScene( this->imagenResultado );
		scn->addPixmap(imagen);
		this->imagenResultado->setScene(scn);
	}else{
		QMessageBox::warning(this, tr("Error!"), tr("Error en el proceso de carga de la imagen resultado."));
	}

}

QList<QImage> panel1::getResultData(){

}

QString panel1::getHelpData(){
	return Constants::Instance()->getPanel1HelpMessage();
}

QString panel1::getExecutionData(){
	QString executionData;
	QString val;
	for(int i=0; i<Controlador::Instance()->getSimpleImageExecution()->size();i++){
		TimeContainer timeContainer;
		timeContainer = Controlador::Instance()->getSimpleImageExecution()->at(i);
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

void panel1::saveImages(){

	if(!this->getMatrixListResult()->isEmpty()){

		QString fileName = QFileDialog::getSaveFileName(this, tr("Guardar imagen"),
										QDir::homePath(),
		                            tr("Images (*.jpg)"));

		if ( fileName.isNull() == false )
		{
			this->getMatrixListResult()->at(0).save(fileName+".jpg","JPG");
		}

	}else{

	}
}

//Obtiene el String de información correspondiente para el panel
QString panel1::getAlgorythmHelpData(){
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

void panel1::showAlgorythmHelpMessage() {

	QMessageBox::about(this, tr("Ayuda"), this->getAlgorythmHelpData());

}
