/*
 * panelList.h
 *
 * Creado: 25/02/2012
 * Autor: jose
 * Descripción: clase encargada de instanciar y manejar el listado de paneles como una entidad
 * diferenciada de manera que se gestione de manera organizada las transiciones entre los mismos.
 */

#ifndef PANELLIST_H
#define PANELLIST_H

#include <QtGui/QStackedWidget>
#include <QtGui/QListWidget>
#include <QtGui/QWidget>
#include <QtGui/QListWidgetItem>
#include <QtGui/QStackedWidget>

class panelList : public QWidget
{
    Q_OBJECT
public:
    panelList();
public slots:
	void verticalMenuClicked(QListWidgetItem *current, QListWidgetItem *previous);

public:
	 QStackedWidget *pagesWidget;
};

#endif	/* PANELLIST_H */

