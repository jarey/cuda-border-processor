/*
 * VideoWidget.h
 *
 * Creado: 09/07/2012
 * Autor: jose
 * Descripción: definición de los elementos que formarán el panel de representación de las imágenes
 * 				recogidas por medio de la cámara web.
 */

#ifndef VIDEOWIDGET_H
#define VIDEOWIDGET_H

#include <QWidget>
#include <QPainter>

class VideoWidget: public QWidget {
Q_OBJECT
public:
	explicit VideoWidget(QWidget *parent = 0);
	QImage img;

protected:
	void paintEvent(QPaintEvent *event);

signals:

public slots:

};

#endif // VIDEOWIDGET_H
