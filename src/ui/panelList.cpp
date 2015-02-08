/*
 * panelList.cpp
 *
 * Creado: 25/02/2012
 * Autor: jose
 * Descripción: clase encargada de instanciar y manejar el listado de paneles como una entidad
 * diferenciada de manera que se gestione de manera organizada las transiciones entre los mismos.
 */

#include <QtGui>
#include "panelList.h"
#include "panel1.h"
#include "Panel2.h"
#include "Panel3.h"
#include "Panel4.h"
#include "Panel5.h"
#include "Panel6.h"
#include "qpushbutton.h"
#include <typeinfo>
#include "../common/Controlador.h"
#include "../common/Constants.h"
#include "./src/cudainfo/czdialog.h"

panelList::panelList() {
	QHBoxLayout *horizontalLayout = new QHBoxLayout();

	this->pagesWidget = new QStackedWidget;

	//Ejecución Simple (1 imagen)
	pagesWidget->addWidget(new panel1());
	//Ejecución múltiple (varias imágenes)
	pagesWidget->addWidget(new Panel2());
	//Ejecución varios algoritmos
	pagesWidget->addWidget(new Panel3());
	//Gráficas 3D
	pagesWidget->addWidget(new Panel4());
	//Webcam
	pagesWidget->addWidget(new Panel6());
	//Información tarjeta gráfica
	pagesWidget->addWidget(new CZDialog());
	//Configuración programa
	pagesWidget->addWidget(new Panel5());

	horizontalLayout->addWidget(pagesWidget);
	this->pagesWidget->setCurrentIndex(0);
	setLayout(horizontalLayout);
}

void panelList::verticalMenuClicked(QListWidgetItem *current,
		QListWidgetItem *previous) {

}
