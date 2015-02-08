/*
 * TimeDialog.cpp
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Clase encargada de implementar la ventana de visualización de tiempos de
 *  			 la aplicación.
 */

#include <QtGui>

#include "TimeDialog.h"
#include "./src/common/Controlador.h"
#include "./src/common/Constants.h"


/*
 * Constructor: TimeDialog
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Constructor del panel, crea los elementos, los informa según la configuración de idioma y carga la información de la ventana
 *  			según el modo de procesamiento que esté informado en el controlador de la aplicación.
 */
TimeDialog::TimeDialog()
{
	this->createElements();
	this->retranslateUi(this);
    setWindowTitle(tr("Ventana de tiempos "));
    this->loadTimeInformation();
}

/*
 * Método: createElements
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Realiza la creación y disposicion de los elementos de la ventana según un determinado layout.
 */
void TimeDialog::createElements()
{
    this->resize(702, 529);
    verticalLayout_3 = new QVBoxLayout(this);
    verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
    verticalLayout_2 = new QVBoxLayout();
    verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
    groupBox = new QGroupBox(this);
    groupBox->setObjectName(QString::fromUtf8("groupBox"));
    verticalLayout = new QVBoxLayout(groupBox);
    verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
    tableWidget = new QTableWidget(groupBox);
    if (tableWidget->columnCount() < 5){
        tableWidget->setColumnCount(5);
    }

    QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
    __qtablewidgetitem->setTextAlignment(Qt::AlignHCenter|Qt::AlignVCenter|Qt::AlignCenter);

    tableWidget->setHorizontalHeaderItem(0, __qtablewidgetitem);
    QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();

    __qtablewidgetitem1->setTextAlignment(Qt::AlignHCenter|Qt::AlignVCenter|Qt::AlignCenter);
    tableWidget->setHorizontalHeaderItem(1, __qtablewidgetitem1);
    QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();

    __qtablewidgetitem2->setTextAlignment(Qt::AlignHCenter|Qt::AlignVCenter|Qt::AlignCenter);
    tableWidget->setHorizontalHeaderItem(2, __qtablewidgetitem2);
    QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();

    __qtablewidgetitem3->setTextAlignment(Qt::AlignHCenter|Qt::AlignVCenter|Qt::AlignCenter);
    tableWidget->setHorizontalHeaderItem(3, __qtablewidgetitem3);
    QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();

    __qtablewidgetitem4->setTextAlignment(Qt::AlignHCenter|Qt::AlignVCenter|Qt::AlignCenter);
    tableWidget->setHorizontalHeaderItem(4, __qtablewidgetitem4);
    tableWidget->setObjectName(QString::fromUtf8("tableWidget"));
    QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    sizePolicy.setHorizontalStretch(20);
    sizePolicy.setVerticalStretch(0);
    sizePolicy.setHeightForWidth(tableWidget->sizePolicy().hasHeightForWidth());
    tableWidget->setSizePolicy(sizePolicy);
    tableWidget->setMinimumSize(QSize(658, 0));
    tableWidget->setGridStyle(Qt::SolidLine);
    tableWidget->setSortingEnabled(false);
    tableWidget->horizontalHeader()->setVisible(true);
    tableWidget->horizontalHeader()->setCascadingSectionResizes(false);
    tableWidget->horizontalHeader()->setDefaultSectionSize(100);
    tableWidget->horizontalHeader()->setMinimumSectionSize(50);
    tableWidget->horizontalHeader()->setProperty("showSortIndicator", QVariant(true));
    tableWidget->horizontalHeader()->setStretchLastSection(true);
    tableWidget->setColumnCount(5);
    verticalLayout->addWidget(tableWidget);
    verticalLayout_2->addWidget(groupBox);
    horizontalLayout_2 = new QHBoxLayout();
    horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
    horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    horizontalLayout_2->addItem(horizontalSpacer);
    pushButton_2 = new QPushButton(this);
    pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
    horizontalLayout_2->addWidget(pushButton_2);
    pushButton = new QPushButton(this);
    pushButton->setObjectName(QString::fromUtf8("pushButton"));
    horizontalLayout_2->addWidget(pushButton);
    horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    horizontalLayout_2->addItem(horizontalSpacer_2);
    verticalLayout_2->addLayout(horizontalLayout_2);
    verticalLayout_3->addLayout(verticalLayout_2);

    tableWidget->horizontalHeader()->resizeSection(0, 120);
   tableWidget->horizontalHeader()->resizeSection(1, 160);
    tableWidget->horizontalHeader()->resizeSection(2, 100);
    tableWidget->horizontalHeader()->resizeSection(3, 100);
    tableWidget->horizontalHeader()->resizeSection(4, 100);
    retranslateUi(this);

    //Al pulsar aceptar se cerrará la ventana de tiempos
    connect(this->pushButton, SIGNAL(clicked()), this, SLOT(accept()));
    connect(this->pushButton_2, SIGNAL(clicked()),this,SLOT(resetInfo()));
    QMetaObject::connectSlotsByName(this);
}

/*
 * Método:retranslateUi
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Realiza la información de los textos de la ventana de acuerddo a la configuración establecida.
 */
void TimeDialog::retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("Dialog", "Tiempos de Ejecuci\303\263n ", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = tableWidget->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("Dialog", "Algoritmo", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = tableWidget->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("Dialog", "Modo Ejecuci\303\263n", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = tableWidget->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("Dialog", "Tiempo CPU", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = tableWidget->horizontalHeaderItem(3);
        ___qtablewidgetitem3->setText(QApplication::translate("Dialog", "Tiempo GPU", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = tableWidget->horizontalHeaderItem(4);
        ___qtablewidgetitem4->setText(QApplication::translate("Dialog", "Aceleraci\303\263n GPU", 0, QApplication::UnicodeUTF8));
        pushButton_2->setText(QApplication::translate("Dialog", "Resetear Info.", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("Dialog", "Aceptar", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

/*
 * Método: loadTimeInformation
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Realiza la carga de información de tiempos en la tabla de la pantalla de acuerdo al modo de ejecución informado en el controlador
 *  			de la aplicación.
 */
void TimeDialog::loadTimeInformation(){
	//Se mira el modo actual de la aplicación para saber desde dónde se ha abierto el panel.
	QString mode = Controlador::Instance()->getApplicationMode();
	int sizeValue = 0;
	QList<TimeContainer>* list;
	//Si la cola que le corresponde mirar tiene resultados se cargan en la tabla.
	if(mode == Constants::Instance()->getSimpleImageMode()){
		sizeValue = Controlador::Instance()->getSimpleImageExecution()->size();
		list = Controlador::Instance()->getSimpleImageExecution();
	}else if(mode == Constants::Instance()->getMultipleImageMode()){
		sizeValue = Controlador::Instance()->getMultipleImageExecution()->size();
		list = Controlador::Instance()->getMultipleImageExecution();
	}else if(mode == Constants::Instance()->getMultipleAlgorythmMode()){
		sizeValue = Controlador::Instance()->getMultipleAlgorythmExecution()->size();
		list = Controlador::Instance()->getMultipleAlgorythmExecution();
	}

	if(sizeValue > 0){
		loadTable(list);
	}else{
		//En caso contrario se muestra un error y se cierra el panel.
		//showMessage();
	}
}

/*
 * Método: loadTable
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Realiza la carga de información de tiempos en la tabla de la ventana.
 */
void TimeDialog::loadTable(QList<TimeContainer>* list){
	QString algorythmName;
	QString executionMode;
	QString cpuTime;
	QString gpuTime;
	QString gpuAcceleration;

	TimeContainer container;

	//Para cada elemento del listado objetivo se crea una entrada en la tabla de tiempos.
	for(int i=0 ; i<list->size() ; i++){
		//QTableWidgetItems para realizar las inserciones en la tabla.
		QTableWidgetItem *item1 = new QTableWidgetItem();
		QTableWidgetItem *item2 = new QTableWidgetItem();
		QTableWidgetItem *item3 = new QTableWidgetItem();
		QTableWidgetItem *item4 = new QTableWidgetItem();
		QTableWidgetItem *item5 = new QTableWidgetItem();

		//Se introduce la nueva fila
		int row = this->tableWidget->rowCount();
		this->tableWidget->insertRow(row);

		//Se recogen los datos
		container = list->at(i);

		algorythmName = container.getProcess();

		if(container.getCPUMilliseconds()!=NULL && container.getCPUMilliseconds()!=0){
			cpuTime = cpuTime.setNum(container.getCPUMilliseconds()) + " ms.";
		}else{
					cpuTime = "-";
		}

		if(container.getGPUMilliseconds()!=NULL && container.getGPUMilliseconds()!=0){
			gpuTime = gpuTime.setNum(container.getGPUMilliseconds()) + " ms.";
		}else{
			gpuTime = "-";
		}

		if(container.getCPUMilliseconds()!=NULL && container.getCPUMilliseconds()!=0 && container.getGPUMilliseconds()!=NULL && container.getGPUMilliseconds()!=0){
			gpuAcceleration = gpuAcceleration.setNum(container.getGraphiAceleration()) + "X";
		}else{
			gpuAcceleration = "-";
		}

		executionMode = Controlador::Instance()->getApplicationMode();

		//Nombre del algortimo
		item1->setText(algorythmName);
		item1->setTextAlignment(Qt::AlignCenter);
		item2->setText(executionMode);
		item2->setTextAlignment(Qt::AlignCenter);
		item3->setText(cpuTime);
		item3->setTextAlignment(Qt::AlignCenter);
		item4->setText(gpuTime);
		item4->setTextAlignment(Qt::AlignCenter);
		item5->setText(gpuAcceleration);
		item5->setTextAlignment(Qt::AlignCenter);

		//Se puebla la fila creada
		this->tableWidget->setItem(row, 0, item1);
		this->tableWidget->setItem(row, 1, item2);
		this->tableWidget->setItem(row, 2, item3);
		this->tableWidget->setItem(row, 3, item4);
		this->tableWidget->setItem(row, 4, item5);
	}
}

/*
 * Método: close
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Realiza el cerrado de la ventana.
 */
void TimeDialog::close(){
	//this->closeEvent();
}

/*
 * Método: resetInfo
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Realiza el borrado de la información de la ventana y de las estructuras de datos asociadas en el controlador de la aplicación.
 */
void TimeDialog::resetInfo(){
	//Se mira el modo actual de la aplicación para saber desde dónde se ha abierto el panel.
	QString mode = Controlador::Instance()->getApplicationMode();

	//Si la cola que le corresponde mirar tiene resultados se cargan en la tabla.
	if(mode == Constants::Instance()->getSimpleImageMode()){
		Controlador::Instance()->getSimpleImageExecution()->clear();
	}else if(mode == Constants::Instance()->getMultipleImageMode()){
		Controlador::Instance()->getMultipleImageExecution()->clear();
	}else if(mode == Constants::Instance()->getMultipleAlgorythmMode()){
		Controlador::Instance()->getMultipleAlgorythmExecution()->clear();
	}

	this->tableWidget->clear();
}
