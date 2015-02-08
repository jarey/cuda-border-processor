/*
 * Panel6.h
 *
 * Creado: 25/02/2012
 * Autor: jose
 * Descripción: Modo WebCam
 */

#ifndef PANEL6_H
#define PANEL6_H

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
#include <QList>
#include "droparea.h"
#include "PanelBase.h"
#include "./src/ui/webcam/VideoWidget.h"
#include "./src/ui/webcam/CaptureThread.h"

class Panel6 : public PanelBase
{
	Q_OBJECT

public:
	Panel6();
	~Panel6();
	//Atributos
		//Layouts empleados para el panel
	QVBoxLayout *verticalLayout1;
	QVBoxLayout *verticalLayout2;
	QHBoxLayout *horizontalLayout1;
	QVBoxLayout *verticalLayout3;
	QHBoxLayout *horizontalLayout2;
	QHBoxLayout *labelComboAlgoritmo;
	QHBoxLayout *labelCheckHisteresis;
	QHBoxLayout *labelSpinnerRadioGaus;
	QHBoxLayout *labelSpinnerSigmaGaus;
	QHBoxLayout *labelSpinnerUmbralSuperior;
	QHBoxLayout *labelSpinnerUmbralInferior;
	QGroupBox *packagesGroup;
	//Etiquetas:
	//Imagen original
	VideoWidget *frameOrig;
	//Imagen resultado
	QLabel *imagenResultadoLabel;
	//Algoritmos
	QLabel *algoritmosLabel;
	//Histeresis
	QLabel *histeresisLabel;
	//Sigma gauss
	QLabel *sigmaGaussLabel;
	//Radio gauss
	QLabel *radioGaussLabel;
	//Umbral superior
	QLabel *umbralSuperiorLabel;
	//Umbral inferior
	QLabel *umbralInferiorLabel;
	//Lienzos para las imágenes origen y resultado.
	DropArea *imagenOriginal;
	QGraphicsView *imagenResultado;
	//Selección de algoritmos
	QComboBox *algorythmSelect;
	//Check de histeresis
	QCheckBox *histeresisCheck;
	//Spinners para valores de umbrales y radio/sigma
	QSpinBox *radioGauss;
	QDoubleSpinBox *sigmaGauss;
	QDoubleSpinBox *umbralSuperior;
	QDoubleSpinBox *umbralInferior;
	QSpacerItem *horizontalSpacer;
	QSpacerItem *horizontalSpacer_2;
	CaptureThread *t;
	bool canSave;
	QPushButton *botonPrueba;

	//Funciones
	void showElements(QString DisplayConstant);
	void loadOriginalImage(QString ruta);
	void loadResultImage(QImage resultImage);
	//Por herencia
	void loadOriginData(QStringList stringList);
	QList<QImage> getOriginData();
	void loadResultData(QList<QImage> *imageList);
	QList<QImage> getResultData();
	QString getHelpData();
	QString getExecutionData();
	void saveImages();
	void emittMessage(QString message);
	void setCanSave(bool value);
	bool getCanSave();
	QString getAlgorythmHelpData();


public slots:
	void getDrop(const QMimeData *mimeData);
	void consultarCombo();
	void showAlgorythmHelpMessage();

private:
	void createComponents();
	void instantiateElements();
	void layoutComponents();
};

#endif /* PANEL6_H_ */
