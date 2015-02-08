/*
 * Panel3.cpp : modo MULTIPLE ALGORITMO
 *
 * Creado: 25/02/2012
 * Autor: jose
 */

#include "Panel3.h"
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
#include "./src/imagefilters/CannyImageFilter.h"
#include "./src/imagefilters/LaplacianOfGaussianImageFilter.h"
#include "./src/imagefilters/ImageFilterFactory.h"

using namespace std;


void Panel3::getDrop(const QMimeData *mimeData) {
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

Panel3::Panel3() {
	//Se le asigna el modo de la aplicación correspondiente al panel
	Constants *constants = Constants::Instance();
	this->setApplicationMode(constants->getMultipleAlgorythmMode());
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

Panel3::~Panel3(){

	delete packagesGroup;
	//Layout horizontal
	delete horizontalLayout1;
	delete horizontalLayout3;
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
	delete verticaLayoutButtons;
	//Se instancia y se configura las columnas de la tabla.
	delete table;
	delete addAlgorythm ;
	delete removeAlgorythm;
	delete filterList;

}

void Panel3::createComponents() {
	//Se instancian y configuran los componentes
	this->instantiateElements();
	//Se les da el layout deseado dentro del panel
	this->layoutComponents();
}

void Panel3::instantiateElements() {

	packagesGroup = new QGroupBox(
			tr("Ejecución de múltiples algoritmos sobre una imagen."));
	//Layout horizontal
	horizontalLayout1 = new QHBoxLayout();
	horizontalLayout3 = new QHBoxLayout();
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
	imagenOriginal->setFixedSize(200, 200);
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
	verticaLayoutButtons = new QVBoxLayout();
	//Se instancia y se configura las columnas de la tabla.
	table = new QTableWidget(0, 6);

	table->setBaseSize(400, 200);

	//Listado de etiquetas
	QStringList labels;
	labels << tr("Algoritmo") << tr("Histeresis") << tr("Radio Gauss") << tr(
			"Sigma Gauss") << tr("Umbral Inferior") << tr("Umbral Superior");
	table->setColumnCount(6);
	table->setHorizontalHeaderLabels(labels);
	//table->horizontalHeader()->setResizeMode(0, QHeaderView::Stretch);
	table->verticalHeader()->hide();
	table->setShowGrid(true);

	//Seteo de tamaño de cada una de las columnas de la cabecera horizontal
	table->horizontalHeader()->resizeSection(0, 204);
	table->horizontalHeader()->resizeSection(1, 165);
	table->horizontalHeader()->resizeSection(2, 135);
	table->horizontalHeader()->resizeSection(3, 135);
	table->horizontalHeader()->resizeSection(4, 135);
	table->horizontalHeader()->resizeSection(5, 135);
	table->setSelectionBehavior(QAbstractItemView::SelectRows);

	addAlgorythm = new QPushButton("Añadir >>");
	addAlgorythm->setIcon(QIcon(":/images/plus-icon.png"));

	removeAlgorythm = new QPushButton("Eliminar X");
	removeAlgorythm->setIcon(QIcon(":/images/remove-icon.png"));
	//Se conecta en el botón de añadir algoritmo y de eliminar algoritmo a sus respectivos slots
	filterList = new QList<ImageFilter*>();
	//Añadir
	connect(addAlgorythm, SIGNAL(pressed()), this, SLOT(addToTable()));
	connect(removeAlgorythm, SIGNAL(pressed()), this, SLOT(RemoveFromTable()));

	botonPrueba = new QPushButton();
	botonPrueba->setStyleSheet("background:transparent;border-radius:5px;;max-width:20px;max-height:20px;min-width:20px;min-height:20px;");
	botonPrueba->setIcon(QIcon(":images/help.png"));

}

void Panel3::layoutComponents() {
	//Posicionamiento de los elementos en los layouts
	//Columnas verticales para label/lienzo
	verticalLayout1->addWidget(imagenOriginalLabel);
	verticalLayout1->setSpacing(1);
	verticalLayout1->addWidget(imagenOriginal);
	horizontalLayout1->addWidget(scrollArea);
	verticalLayout2->addWidget(imagenResultadoLabel);
	verticalLayout2->setSpacing(1);
	verticalLayout2->addWidget(imagenResultado);
	//Se le añade la parte de la imagen original y su etiqueta:
	horizontalLayout3->addLayout(verticalLayout1);
	//Se le añaden los dos botones y la tabla
	//Para los botones habría que barajar un vertical layout
	verticaLayoutButtons->addWidget(addAlgorythm);
	verticaLayoutButtons->addWidget(removeAlgorythm);
	horizontalLayout3->addLayout(verticaLayoutButtons);
	horizontalLayout3->addWidget(table);
	horizontalLayout3->SetMinimumSize;

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
	verticalLayout3->addLayout(horizontalLayout3);
	verticalLayout3->addLayout(horizontalLayout1);
	verticalLayout3->setAlignment(Qt::AlignLeft);
	//Seteo del layout final grupo
	packagesGroup->setLayout(verticalLayout3);
	//Creo un layout más para añadir el grupo a este layout
	QVBoxLayout *verticalLayoutAux = new QVBoxLayout();
	verticalLayoutAux->addWidget(packagesGroup);
	setLayout(verticalLayoutAux);
}

void Panel3::consultarCombo() {
	Controlador *controlador = Controlador::Instance();
	controlador->setAlgorythmSelected(algorythmSelect->currentText());
	showElements(algorythmSelect->currentText());
}

void Panel3::showElements(QString displayMode) {
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

void Panel3::loadOriginalImages(QStringList rutas) {
	if (!rutas.isEmpty()) {
		Controlador *controlador = Controlador::Instance();
		QHBoxLayout *horizontalLayout = new QHBoxLayout();
		QWidget *widget = new QWidget();
		for (int i = 0; i < rutas.size(); ++i) {
			QHBoxLayout *horizontalLayout1 = new QHBoxLayout();
			QGraphicsView *qImage = new QGraphicsView();
			QPixmap imagen = QPixmap(rutas.at(i));
			QGraphicsScene *scn = new QGraphicsScene(this->imagenOriginal);
			scn->addPixmap(imagen);
			qImage->setScene(scn);
			horizontalLayout1->addWidget(qImage);
			controlador->getMatrixListOrigin()->append(imagen.toImage());
			horizontalLayout->addLayout(horizontalLayout1);
		}
		widget->setLayout(horizontalLayout);
		this->scrollArea->setWidget(widget);
	}
}
void Panel3::loadResultImage(QImage resultImage){
}


void Panel3::loadOriginData(QStringList list) {
	QString ruta = list.at(0);
	if(!ruta.isNull())
		{
		this->getMatrixListOrigin()->clear();
		this->getMatrixListResult()->clear();
			Controlador *controlador = Controlador::Instance();

			QPixmap imagen =QPixmap(ruta);
			//Se setea la imagen cargada en el controlador.
			controlador->setOriginImage(imagen.toImage());
			controlador->getMatrixListOrigin()->append(imagen.toImage());
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
			this->getMatrixListOrigin()->append(imagen3);
			QWidget *widget = new QWidget();
			scrollArea->setWidget(widget);
		}

}

QList<QImage> Panel3::getOriginData() {

}

void Panel3::loadResultData(QList<QImage> *imageList) {
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
					if (imagen.height() > scrollArea->height() - 10 || imagen.width()
							> scrollArea->height() - 10) {
						scaledImage = imagen.scaledToHeight(scrollArea->height() - 10,
								Qt::FastTransformation);

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
			}
}

QList<QImage> Panel3::getResultData() {

}

void Panel3::addToTable() {

	//Se mapean los datos del algoritmo en ese instante y
	//se vuelvan a la tabla comprobando de si ya existe ese algoritmo en ella.
	QString algorythmName;
	QString histeresisValue = "-";
	QString radioGaussValue = "-";
	QString sigmaGaussValue = "-";
	QString umbralInferior = "-";
	QString umbralSuperior = "-";

	//Mapeamos los datos de pantalla a variables
	int radioGauss = this->radioGauss->value();
	double sigmaGauss = this->sigmaGauss->value();
	float lowerThreshold = this->umbralInferior->value();
	float higherThreshold = this->umbralSuperior->value();
	bool histeresis = this->histeresisCheck->isChecked();
	algorythmName = this->algorythmSelect->currentText();

	bool exists = false;
	ImageFilter *fil = new ImageFilter;
	//Se realiza búsqueda previa para saber si se debe permitir la inserción del nuevo algoritmo o no.
	for (int i = 0; i < this->filterList->size(); i++) {
		//Se comparan los nombres de los algoritmos
		fil = this->filterList->at(i);

		if (algorythmName == fil->getFilterName()) {

			if (algorythmName == Constants::Instance()->getCannyConstant()) {
				//En el caso de que sea canny o laplacian of gaussian se deben comapra todos los atributos de los filtros apra saber
				//si ya hay insertado un filtro con las mismas características exactamente, si no tiene todos los atributos iguales
				//permitiremos igualmente realizar la inserción.

				//cast a canny
				CannyImageFilter *cannyTempFilter = dynamic_cast<CannyImageFilter*> (fil);

				if (algorythmName == cannyTempFilter->getFilterName()
						&& radioGauss == cannyTempFilter->getRadioGauss()
						&& sigmaGauss == cannyTempFilter->getSigmaGauss()
						&& lowerThreshold
								== cannyTempFilter->getLowerThreshold()
						&& higherThreshold
								== cannyTempFilter->getHigherThreshold()
						&& histeresis == cannyTempFilter->getHisteresis()) {

					exists = true;
				}
			} else if (algorythmName
					== Constants::Instance()->getLaplacianOfGaussianConstant()) {
				//cast a laplacian of gaussian
				LaplacianOfGaussianImageFilter *laplacianOfGaussianTempFilter = dynamic_cast<LaplacianOfGaussianImageFilter*> (fil);

				if (algorythmName == laplacianOfGaussianTempFilter->getFilterName()
						&& radioGauss == laplacianOfGaussianTempFilter->getRadioGauss()
						&& sigmaGauss == laplacianOfGaussianTempFilter->getSigmaGauss()
						) {

					exists = true;
				}
			} else {
				exists = true;
			}
		}
	}

	if(!exists){
	//Se realiza una instanciación del filtro concreto a través de la factoría de filtros.
	ImageFilterFactory fac = (ImageFilterFactory) *new ImageFilterFactory;
	ImageFilter* imageFilter = fac.getImageFilterInstance(algorythmName, histeresis,radioGauss, sigmaGauss, lowerThreshold, higherThreshold);

	//Se tiene en cuenta si es canny para determinar el Si o el NO en para la histeresis en la tabla.
	if (algorythmName == Constants::Instance()->getCannyConstant()) {
		if (this->histeresisCheck->isEnabled()) {
			if (this->histeresisCheck->isChecked()) {
				histeresisValue = "Sí";
			} else {
				histeresisValue = "No";
			}
		}
	}

	//Se tiene en cuenta si es canny o laplacianofgaussian para incluir los valores concretos de los parámetros en la tabla.
	if(algorythmName == Constants::Instance()->getCannyConstant() || algorythmName == Constants::Instance()->getLaplacianOfGaussianConstant()){
		//Pasamos los datos de pantalla en sus respectiv tipos de datos
		//a strings que se mostrarán en la tabla.
		radioGaussValue = radioGaussValue.setNum(radioGauss);
		sigmaGaussValue = sigmaGaussValue.setNum(sigmaGauss);
		umbralInferior = umbralInferior.setNum(lowerThreshold);
		umbralSuperior = umbralSuperior.setNum(higherThreshold);
	}

	//QTableWidgetItems para realizar las inserciones en la tabla.
	QTableWidgetItem *item1 = new QTableWidgetItem();
	QTableWidgetItem *item2 = new QTableWidgetItem();
	QTableWidgetItem *item3 = new QTableWidgetItem();
	QTableWidgetItem *item4 = new QTableWidgetItem();
	QTableWidgetItem *item5 = new QTableWidgetItem();
	QTableWidgetItem *item6 = new QTableWidgetItem();

	item1->setText(algorythmName);
	int row = this->table->rowCount();
	//Nombre del algortimo
	this->table->insertRow(row);
	this->table->setItem(row, 0, item1);
	//Histeresis
	item2->setText(histeresisValue);
	this->table->setItem(row, 1, item2);
	//Radio gauss
	item3->setText(radioGaussValue);
	this->table->setItem(row, 2, item3);
	//Sigma gauss
	item4->setText(sigmaGaussValue);
	this->table->setItem(row, 3, item4);
	//Umbral inferior
	item5->setText(umbralInferior);
	this->table->setItem(row, 4, item5);
	//Umbral superior
	item6->setText(umbralSuperior);
	this->table->setItem(row, 5, item6);

	//algorythmList	++
	filterList->append(imageFilter);

	}else{
		QMessageBox::warning(
									this,
									tr("¡Aviso!"),
									tr(
											"La configuración seleccionada ya está presente dentro de la tabla de algoritmos para ejecución."));
	}
}

void Panel3::RemoveFromTable() {
	//Si se tiene algún algoritmo seleccionado en la tabla se lanza la ventana de confirmación de borrado y se elimina si acepta.

	int row = -1;
	row = this->table->currentRow();

	if(row>=0){

			QMessageBox::StandardButton ret;
			ret = QMessageBox::warning(this, tr("¡Confirmación!"),
					tr("¿Está seguro de que desea eliminar el algoritmo seleccionado de la tabla de algoritmos para ejecución?"),
					QMessageBox::Yes | QMessageBox::No);
			if (ret == QMessageBox::Yes){
				this->table->removeRow(row);
				filterList->removeAt(row);
			}
	}else{
		QMessageBox::warning(
							this,
							tr("¡Aviso!"),
							tr(
									"No existe ningún elemento seleccionado, debe seleccionar un algoritmo de la tabla para poder proceder a su eliminación."));
	}
}

QList<ImageFilter*> *Panel3::getFilterList(){
	return this->filterList;
}

QString Panel3::getHelpData(){
	return Constants::Instance()->getPanel3HelpMessage();
}

QString Panel3::getExecutionData(){
	QString executionData;
	QString val;
	for(int i=0; i<Controlador::Instance()->getMultipleAlgorythmExecution()->size();i++){
		TimeContainer timeContainer;
		timeContainer = Controlador::Instance()->getMultipleAlgorythmExecution()->at(i);
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
		}else if(timeContainer.getExecutionType() == "CPU vs GPU"){
			//Caso para CPU vs GPU
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


void Panel3::saveImages(){
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
QString Panel3::getAlgorythmHelpData(){
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

void Panel3::showAlgorythmHelpMessage() {

	QMessageBox::about(this, tr("Ayuda"), this->getAlgorythmHelpData());

}
