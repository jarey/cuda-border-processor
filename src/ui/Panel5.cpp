/*
 * Panel5.cpp : Panel De Configuración.
 *
 * Creado: 12/06/2012
 * Autor: jose
 */

#include "Panel5.h"
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
#include "./src/cudainfo/czdialog.h"

void Panel5::getDrop(const QMimeData *mimeData){
}

Panel5::Panel5() {
	//Se le asigna el modo de la aplicación correspondiente al panel
	Constants *constants = Constants::Instance();
	this->setApplicationMode(constants->getPlotMode());
	//Se instancian y se les da layout a los componentes
	createComponents();
	//Se cargan las tarjetas gráficas disponibles en el combo
	this->loadGPUS();

	//Se setean los valores de control de habilitación de los botones de acciones en pantalla.
	setRunButtonNeeded(false);
	setStopButtonNeeded(false);
	setTimeButtonNeeded(false);
	setOpenButtonNeeded(false);
	setSaveButtonNeeded(false);

	connect(this->pushButton, SIGNAL(pressed()), this, SLOT(saveData()));

}

Panel5::~Panel5(){
	//Posicionamiento de los elementos en los layouts
	delete groupBoxWidget;
	delete verticalLayout_4;
	delete verticalSpacer;
	delete horizontalLayout_2;
	delete horizontalLayout;
	delete groupBox_2;
	delete verticalLayout_3;
	delete verticalSpacer_5;
	delete comboBox;
	delete verticalSpacer_6;
	delete verticalLayout_2;
	delete groupBox;
	delete verticalLayout;
	delete radioButton;
	delete verticalSpacer_3;
	delete radioButton_2;
	delete verticalSpacer_4;
	delete radioButton_3;
	delete horizontalSpacer;
	delete verticalSpacer_2;
	delete pushButton;
	delete verticalLayout_5;
}

void Panel5::createComponents() {
	//Se instancian y configuran los componentes
	this->instantiateElements();
	//Se les da el layout deseado dentro del panel
	this->layoutComponents();
	this->manageElements(Controlador::Instance()->getIsCudaCapable());
}

void Panel5::instantiateElements() {
}

void Panel5::layoutComponents() {
	//Posicionamiento de los elementos en los layouts
	groupBoxWidget = new QGroupBox("Configuración de programa.");
	verticalLayout_4 = new QVBoxLayout();
    verticalLayout_4->setSpacing(10);
    verticalLayout_4->setContentsMargins(11, 11, 11, 11);
    verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
    verticalSpacer = new QSpacerItem(20, 108, QSizePolicy::Minimum, QSizePolicy::Expanding);

    verticalLayout_4->addItem(verticalSpacer);
    horizontalLayout_2 = new QHBoxLayout();
    horizontalLayout_2->setSpacing(6);
    horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
    horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    horizontalLayout_2->addItem(horizontalSpacer_2);
    horizontalLayout = new QHBoxLayout();
    horizontalLayout->setSpacing(6);
    horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
    groupBox_2 = new QGroupBox("Tarjeta gráfica a emplear");
    groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
    verticalLayout_3 = new QVBoxLayout(groupBox_2);
    verticalLayout_3->setSpacing(6);
    verticalLayout_3->setContentsMargins(11, 11, 11, 11);
    verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
    verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

    verticalLayout_3->addItem(verticalSpacer_5);
    comboBox = new QComboBox(groupBox_2);
    comboBox->setObjectName(QString::fromUtf8("comboBox"));
    comboBox->setMinimumSize(QSize(200, 30));

    verticalLayout_3->addWidget(comboBox);

    verticalSpacer_6 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

    verticalLayout_3->addItem(verticalSpacer_6);


    horizontalLayout->addWidget(groupBox_2);

    groupBox = new QGroupBox("Modo de ejecución");
    groupBox->setObjectName(QString::fromUtf8("groupBox"));
    verticalLayout_2 = new QVBoxLayout(groupBox);
    verticalLayout_2->setSpacing(6);
    verticalLayout_2->setContentsMargins(11, 11, 11, 11);
    verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
    verticalLayout = new QVBoxLayout();
    verticalLayout->setSpacing(6);
    verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
    radioButton = new QRadioButton(groupBox);
    radioButton->setText("Ejecución CPU");
    //Se marca por defecto este radio (debería hacerse en función a los valores de datos del controlador)
    radioButton->setChecked(true);
    radioButton->setObjectName(QString::fromUtf8("radioButton"));

    verticalLayout->addWidget(radioButton);

    verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

    verticalLayout->addItem(verticalSpacer_3);

    radioButton_2 = new QRadioButton(groupBox);
    radioButton_2->setText("Ejecución GPU");
    radioButton_2->setObjectName(QString::fromUtf8("radioButton_2"));

    verticalLayout->addWidget(radioButton_2);

    verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

    verticalLayout->addItem(verticalSpacer_4);

    radioButton_3 = new QRadioButton(groupBox);
    radioButton_3->setText("Comparación CPU vs GPU");
    radioButton_3->setObjectName(QString::fromUtf8("radioButton_3"));

    verticalLayout->addWidget(radioButton_3);

    verticalLayout_2->addLayout(verticalLayout);

    horizontalLayout->addWidget(groupBox);

    horizontalLayout_2->addLayout(horizontalLayout);

    horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    horizontalLayout_2->addItem(horizontalSpacer);

    verticalLayout_4->addLayout(horizontalLayout_2);

    verticalSpacer_2 = new QSpacerItem(20, 108, QSizePolicy::Minimum, QSizePolicy::Expanding);

    verticalLayout_4->addItem(verticalSpacer_2);

    pushButton = new QPushButton("Guardar cambios");
    pushButton->setObjectName(QString::fromUtf8("pushButton"));
    pushButton->setMaximumSize(QSize(130, 30));
    pushButton->setLayoutDirection(Qt::LeftToRight);

    verticalLayout_4->addWidget(pushButton);
    groupBoxWidget->setLayout(verticalLayout_4);
    verticalLayout_5 = new QVBoxLayout();
    verticalLayout_5->addWidget(groupBoxWidget);

	setLayout(verticalLayout_5);
}

void Panel5::consultarCombo()
{
}

void Panel5::loadOriginalImage(QString ruta){

}
void Panel5::loadResultImage(QImage resultImage){
}

void Panel5::loadOriginData(QStringList list){
}

QList<QImage> Panel5::getOriginData(){
}

void Panel5::loadResultData(QList<QImage> *imageList){
}


QList<QImage> Panel5::getResultData(){
}

//Método para cargar las tarjetas
void  Panel5::loadGPUS() {
	CZDialog *cudainfo = new CZDialog();
	QStringList nameList;
	nameList=cudainfo->returnDevicesNames();
	this->comboBox->addItems(nameList);
}

//Método para guardar y conectar al botón de guardado.
void  Panel5::saveData() {
	Controlador *controlador = Controlador::Instance();
	//Se obtiene el índice de la tarjeta seleccionada actualmente.
	this->comboBox->currentIndex();

	//Se obtiene el check marcado actualmente.
	if(this->radioButton->isChecked()){
		controlador->setIsGpuMode(0);
	}else if(this->radioButton_2->isChecked()){
		controlador->setIsGpuMode(1);
	}else if(this->radioButton_3->isChecked()){
		controlador->setIsGpuMode(2);
	}
}

QString  Panel5::getHelpData(){
	return Constants::Instance()->getPanel5HelpMessage();
}

QString Panel5::getExecutionData(){
}

void Panel5::manageElements(unsigned char value){
	//En función del valor pasado como parámetro se habilitan o no los radios y el combo de la pantalla.
	if(value==0){
		this->comboBox->setEnabled(false);
		this->radioButton->setEnabled(false);
		this->radioButton_2->setEnabled(false);
		this->radioButton_3->setEnabled(false);
		this->pushButton->setEnabled(false);
	}else{
		this->comboBox->setEnabled(true);
		this->radioButton->setEnabled(true);
		this->radioButton_2->setEnabled(true);
		this->radioButton_3->setEnabled(true);
		this->pushButton->setEnabled(true);
	}
}

void Panel5::saveImages(){
	this->saveData();
}

//Obtiene el String de información correspondiente para el panel
QString Panel5::getAlgorythmHelpData(){

}

void Panel5::showAlgorythmHelpMessage() {

}
