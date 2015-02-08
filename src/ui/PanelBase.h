/*
 * PanelBase.h
 * Creado: 25/02/2012
 * Autor: jose
 * Descripción: panel con la funcionalidad base para los paneles concretos. Establece los mñetodos
 * abstractos que tendrán que ser obligatoriamente implementados por las subclases de cada panel concreto.
 */

#ifndef PANELBASE_H
#define PANELBASE_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QGraphicsView>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QStatusBar>
#include <QtGui/QTextEdit>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include <QtGui/QGroupBox>
#include "droparea.h"

class PanelBase : public QWidget
{
	Q_OBJECT

public:
	PanelBase();
public:
	//Métodos públicos virtuales cuya implementación correrá a cargo de las clases hijas
	/**
	 * Método que carga la imagen/imágenes origen en el listado dell panel.
	 * @param QStringList listado de string con las urls de las imágenes a cargar
	 * @retunr void
	 */
	void virtual loadOriginData(QStringList stringList)=0;
	/**
	 * Método que devuelve el listado de imágenes origen que posee el panel
	 * @return QList<Qimage> listado de imágenes origen que posee el panel
	 */
	QList<QImage> virtual getOriginData()=0;
	/**
	 * Método que carga la imagen/imágenes procesadas en el listado del panel.
	 * @param QStringList listado de string con las urls de las imágenes a cargar
	 * @retunr void
	 */
	void virtual loadResultData(QList<QImage> *imageList)=0;
	/**
	  * Método que devuelve el listado de imágenes resultado que posee el panel
	  * @return QList<Qimage> listado de imágenes origen que posee el panel
	  */
	QList<QImage> virtual getResultData()=0;

	//Métodos públicos virtuales cuya implementación correrá a cargo de las clases hijas
	/**
	 * Método que carga la imagen/imágenes origen en el listado dell panel.
	 * @param QStringList listado de string con las urls de las imágenes a cargar
	 * @retunr void
	 */
	void virtual saveImages()=0;

	QString getApplicationMode();
	void setApplicationMode(QString mode);
	//Obtiene el String de información correspondiente para el panel
	QString virtual getHelpData()=0;
	//Obtiene el String para mostrar la información de ejecución según el modo
	QString virtual getExecutionData()=0;
	//Obtiene el String de información correspondiente para el panel
	QString virtual getAlgorythmHelpData()=0;

	//Setters
	void setRunButtonNeeded(bool value);
	void setStopButtonNeeded(bool value);
	void setOpenButtonNeeded(bool value);
	void setTimeButtonNeeded(bool value);
	void setSaveButtonNeeded(bool value);
	void setMatrixListOrigin(QList<QImage>* matrixListOrigin);
	void setMatrixListResult(QList<QImage>* matrixListResult);

	//Getters
	bool getRunButtonNeeded();
	bool getStopButtonNeeded();
	bool getOpenButtonNeeded();
	bool getTimeButtonNeeded();
	bool getSaveButtonNeeded();
	QList<QImage>* getMatrixListOrigin();
	QList<QImage>* getMatrixListResult();

private:
	//Atributos privados de la clase.
	/**
	 * Listado de imágenes origen
	 */
	QList<QImage> originalImageList;
	/**
	 * Listado de imágenes procesadas.
	 */
	QList<QImage> resultImageList;

	/**
	 * Modo de la aplicación asociado al panel.
	 */
	QString applicationMode;

	//Métodos privados virtuales a implementar por las clases hijas.
	void virtual createComponents()=0;
	void virtual instantiateElements()=0;
	void virtual layoutComponents()=0;

	bool runButtonNeeded;
	bool stopButtonNeeded;
	bool openButtonNeeded;
	bool timeButtonNeeded;
	bool saveButtonNeeded;

	//Listado de imágenes origen del panel.
	QList<QImage> *matrixListOrigin;

	//Listado de imágenes procesadas del panel.
	QList<QImage> *matrixListResult;

};

#endif /* PANELBASE_H_ */
