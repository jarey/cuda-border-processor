/*
 * Panel5.h
 *
 * Creado: 12/06/2012
 * Autor: jose
 * Descripción: Panel De Configuración.
 */

#ifndef PANEL5_H_
#define PANEL5_H_

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
#include <QtGui/QRadioButton>
#include <QList>
#include "droparea.h"
#include "PanelBase.h"


class Panel5 : public PanelBase
{
	Q_OBJECT

public:
	Panel5();
	~Panel5();
	//Atributos
	QVBoxLayout *verticalLayout_4;
	QSpacerItem *verticalSpacer;
	QHBoxLayout *horizontalLayout_2;
	QSpacerItem *horizontalSpacer_2;
	QHBoxLayout *horizontalLayout;
	QGroupBox *groupBox_2;
	QVBoxLayout *verticalLayout_3;
	QSpacerItem *verticalSpacer_5;
	QComboBox *comboBox;
	QSpacerItem *verticalSpacer_6;
	QGroupBox *groupBox;
	QVBoxLayout *verticalLayout_2;
	QVBoxLayout *verticalLayout;
	QRadioButton *radioButton;
	QSpacerItem *verticalSpacer_3;
	QRadioButton *radioButton_2;
	QSpacerItem *verticalSpacer_4;
	QRadioButton *radioButton_3;
	QSpacerItem *horizontalSpacer;
	QSpacerItem *verticalSpacer_2;
	QPushButton *pushButton;
	QGroupBox *groupBoxWidget;
	QVBoxLayout *verticalLayout_5;
	QPushButton *botonPrueba;

	//Funciones
	void loadOriginalImage(QString ruta);
	void loadResultImage(QImage resultImage);
	//Por herencia
	void loadOriginData(QStringList stringList);
	QList<QImage> getOriginData();
	void loadResultData(QList<QImage> *imageList);
	QList<QImage> getResultData();
	void loadGPUS();
	QString getHelpData();
	QString getExecutionData();
	void saveImages();
	QString getAlgorythmHelpData();


public slots:
	void getDrop(const QMimeData *mimeData);
	void consultarCombo();
	void saveData();
	void showAlgorythmHelpMessage();

private:
	void createComponents();
	void instantiateElements();
	void layoutComponents();
	void manageElements(unsigned char value);
};

#endif /* PANEL5_H_ */
