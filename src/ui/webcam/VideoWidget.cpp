/*
 * VideoWidget.cpp
 *
 * Creado: 09/07/2012
 * Autor: jose
 * Descripción: clase de implementación del panel que reflejará las imágenes recogidas por
 * 				medio de la cámara web.
 */

#include "VideoWidget.h"

VideoWidget::VideoWidget(QWidget *parent) :
	QWidget(parent) {
	setAutoFillBackground(true);
}

void VideoWidget::paintEvent(QPaintEvent *event) {
	try {
		QPainter painter(this);
		painter.setPen(Qt::blue);
		painter.setFont(QFont("Arial", 12));
		painter.drawText(rect(), Qt::AlignCenter,
				"Pulse 'play' en el menú lateral para iniciar la cámara web.");
		QImage image = QImage(":images/webcamBig.png");
		QRect re = QRect(200, 0, 225, 225);
		painter.drawImage(re, image);

		if (!img.isNull()) {
			painter.drawImage(QPoint(0, 0), img);
		}
	} catch (...) {
	}
}
