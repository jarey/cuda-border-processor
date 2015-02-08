/*
 * TimeDialog.cpp
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Clase encargada de realizar la definición de la pantalla emergente de visualización
 *  			de tiempos de ejecución de los procesos de la aplicación.
 */

#ifndef TIMEDIALOG_H
#define TIMEDIALOG_H

#include <QDialog>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QTableWidget>
#include <QtGui/QVBoxLayout>
#include "./src/common/TimeContainer.h"

class TimeDialog : public QDialog
{
    Q_OBJECT

public:
    TimeDialog();

private:
    void createElements();
    void retranslateUi(QDialog *Dialog);

    //Atributos privados del dialog
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QTableWidget *tableWidget;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton_2;
    QPushButton *pushButton;
    QSpacerItem *horizontalSpacer_2;

    //método para cargar información de tiempos según modo.
    void loadTimeInformation();
    //Método de realización de carga de tabla con la información-
    void loadTable(QList<TimeContainer>* list);
private slots:
    void close();
    void resetInfo();
};

#endif
