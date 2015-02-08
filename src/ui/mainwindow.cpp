/*
 * mainwindow.cpp
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: ventana principal de la aplicación. Realiza la gestión de las invocaciones a negocio.
 *  La carga de imágenes, el guardado, la habilitación y deshabilitación de botones y la gestión
 *  de los paneles.
 */

#include "mainwindow.h"
#include <QtGui>
#include <QListWidgetItem>
#include <QWidget>
#include <./src/ui/utils/manhattanstyle.h>  //"manhattanstyle.h"
#include <./src/ui/utils/minisplitter.h>  //"minisplitter.h"
#include "./src/imagefilters/ImageFilter.h"
#include "./src/imagefilters/SobelImageFilter.h"
#include "./src/filterexecutors/FilterExecutorFactory.h"
#include "./src/filterexecutors/FilterExecutor.h"
#include "./src/filterexecutors/SobelFilterExecutor.h"
#include <typeinfo>
#include <QImage>
#include <QPixmap>
#include "./src/imagefilters/ImageProcessingBusiness.h"
#include "./src/common/Constants.h"
#include <QDebug>
#include "./src/ui/PanelBase.h"
#include "./src/ui/panel1.h"
#include "./src/ui/Panel2.h"
#include "./src/ui/Panel3.h"
#include "./src/ui/Panel4.h"
#include "./src/ui/Panel5.h"
#include "./src/ui/Panel6.h"
#include "./src/ui/timedialog/TimeDialog.h"
#include "./src/cudainfo/czdialog.h"
#include "./src/canny/CImage.h"
#include "./src/imagefilters/ImageFilterFactory.h"
#include "./src/imagefilters/LaplacianOfGaussianImageFilter.h"
#include <QMatrix>
#include <QMetaObject>
#include <QStringList>
//Librerias v4l para detección de estado de dispositivo de grabación.
#include <linux/videodev2.h>
#include "libv4l2.h"
#include "libv4lconvert.h"

using namespace ::std;

MainWindow::MainWindow() {

	qApp->setStyle(new ManhattanStyle(QApplication::style()->objectName()));
	//Se instancia el controlador
	controlador = Controlador::Instance();

	textEdit = new QPlainTextEdit;
	createActions();
	setupButtons();
	setupTabBar();
	setupInferiorMenu();
	barraLateral->setCurrentIndex(0);
	setCentralWidget(barraLateral);
	createMenus();
	createToolBars();
	readSettings();
	setCurrentFile("");
	setDockNestingEnabled(false);
	setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
	setCorner(Qt::BottomRightCorner, Qt::BottomDockWidgetArea);
	statusBar()->setProperty("p_styled", true);
	setAcceptDrops(true);
	statusBar()->hide();
	setUnifiedTitleAndToolBarOnMac(true);

	connect(this->barraLateral->m_tabBar, SIGNAL(currentChanged(int)), this,
			SLOT(panelChanged()));
}

void MainWindow::closeEvent(QCloseEvent *event) {
	QWidget *widget0 =this->barraLateral->listaPaneles->pagesWidget->widget(0);
	QWidget *widget1 =this->barraLateral->listaPaneles->pagesWidget->widget(1);
	QWidget *widget2 =this->barraLateral->listaPaneles->pagesWidget->widget(2);
	QWidget *widget3 =this->barraLateral->listaPaneles->pagesWidget->widget(3);
	QWidget *widget4 =this->barraLateral->listaPaneles->pagesWidget->widget(4);
	QWidget *widget5 =this->barraLateral->listaPaneles->pagesWidget->widget(5);
	QWidget *widget6 =this->barraLateral->listaPaneles->pagesWidget->widget(6);

	panel1 *panelImagen0 = dynamic_cast<panel1 *> (widget0);
	Panel2 *panelImagen1 = dynamic_cast<Panel2 *> (widget1);
	Panel3 *panelImagen2 = dynamic_cast<Panel3 *> (widget2);
	Panel4 *panelImagen3 = dynamic_cast<Panel4 *> (widget3);
	Panel5 *panelImagen4 = dynamic_cast<Panel5 *> (widget4);
	CZDialog *panelImagen5 = dynamic_cast<CZDialog *> (widget5);
	Panel6 *panelImagen6 = dynamic_cast<Panel6 *> (widget6);

	delete panelImagen0;
	delete panelImagen1;
	delete panelImagen2;
	delete panelImagen3;
	delete panelImagen4;
	delete panelImagen5;
	delete panelImagen6;
	delete this;
}

void MainWindow::open() {

	Controlador *controlador = Controlador::Instance();
	Constants *constants = Constants::Instance();
	QStringList fileNames;

		//Se debe evaluar el modo de la aplicación para realizar la carga de imagen/imágenes
		//para inicializar el qFileDialog.
		if (controlador->getApplicationMode()
				== constants->getSimpleImageMode()
				|| controlador->getApplicationMode()
						== constants->getMultipleAlgorythmMode()
				|| controlador->getApplicationMode()
						== constants->getPlotMode()) {
			QString fileName = QFileDialog::getOpenFileName(this,
					tr("Abrir imágen"), QDir::homePath(),
					tr("Image Files (*.png *.jpg *.bmp *.jpeg *.gif)"));
			if (!fileName.isEmpty()) {
				fileNames.append(fileName);
			}
		} else if (controlador->getApplicationMode()
				== constants->getMultipleImageMode()) {

			fileNames = QFileDialog::getOpenFileNames(this, tr("Abrir imágenes"),
					QDir::homePath(),
					tr("Image Files (*.png *.jpg *.bmp *.jpeg *.gif)"));
		}

		if (!fileNames.isEmpty()) {

			QWidget
					*widget =
							this->barraLateral->listaPaneles->pagesWidget->currentWidget();
			PanelBase *panel = dynamic_cast<PanelBase *> (widget);
			panel->loadOriginData(fileNames);
			panel->getMatrixListResult()->clear();
			this->enableInfoButton();
			this->enableSaveButton();

		}
}

bool MainWindow::save() {
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();
	PanelBase *panel = dynamic_cast<PanelBase *> (widget);
	panel->saveImages();
}

void MainWindow::about() {
	QMessageBox::about(
			this,
			tr("Sobre la aplicación"),
			tr(
					"Esta aplicación pretende servir para realizar una comparación práctica entre implementaciones realizadas<br/>"
					"para algoritmo de procesamiento de bordes de imágenes en la CPU y en la GPU.<br/>"
					"La aplicación, permite realizar diferentes acciones a lo largo de distintos modos de ejecución, "
					"en los que el usuario puede hacer uso de los algoritmos o llevar a cabo funciones de visualización de información"
					"y/o configuración. El conjunto de modos es el siguiente:<br/>"
					"- <b>Modo ejecución simple.</b><br/>"
					"- <b>Modo ejecución  múltiples algoritmos.</b><br/>"
					"- <b>Modo ejecución múltiples imágenes.</b><br/>"
					"- <b>Modo gráfica de gradiente.</b><br/>"
					"- <b>Modo gráfica.</b><br/>"
					"- <b>Modo información GPU.</b><br/>"
					"- <b>Modo congifuración.</b><br/><br/>"
					"Una descripción más extensa de la funcionalidad proporcionada por cada modo de la aplicación  así como su funcionamiento<br/>"
					"puede ser vista pulsando el botón de ayuda de la parte inferior izquierda de la pantalla, en la barra de iconos<br/>"));
}

void MainWindow::createActions() {
	executeAct = new QAction(QIcon(":/images/play.png"), tr("&Ejecutar"), this);
	executeAct->setStatusTip(tr("Realiza la ejecución"));
	connect(executeAct, SIGNAL(triggered()), this, SLOT(execute()));

	openAct = new QAction(QIcon(":/images/open.png"), tr("&Abrir..."), this);
	openAct->setShortcuts(QKeySequence::Open);
	openAct->setStatusTip(tr("Abrir un fichero de imagen existente"));
	connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

	saveAct = new QAction(QIcon(":/images/save.png"), tr("&Guardar"), this);
	saveAct->setShortcuts(QKeySequence::Save);
	saveAct->setStatusTip(tr("Guardar la imagen(es) a disco"));
	connect(saveAct, SIGNAL(triggered()), this, SLOT(save()));

	exitAct = new QAction(tr("Salir"), this);
	exitAct->setShortcuts(QKeySequence::Quit);

	exitAct->setStatusTip(tr("Salir de la aplicación"));
	connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

	cutAct = new QAction(QIcon(":/images/cut.png"), tr("Cu&t"), this);

	cutAct->setShortcuts(QKeySequence::Cut);
	cutAct->setStatusTip(tr("Cut the current selection's contents to the "
		"clipboard"));
	connect(cutAct, SIGNAL(triggered()), textEdit, SLOT(cut()));

	copyAct = new QAction(QIcon(":/images/copy.png"), tr("&Copy"), this);
	copyAct->setShortcuts(QKeySequence::Copy);
	copyAct->setStatusTip(tr("Copy the current selection's contents to the "
		"clipboard"));
	connect(copyAct, SIGNAL(triggered()), textEdit, SLOT(copy()));

	pasteAct = new QAction(QIcon(":/images/paste.png"), tr("&Paste"), this);
	pasteAct->setShortcuts(QKeySequence::Paste);
	pasteAct->setStatusTip(
			tr("Paste the clipboard's contents into the current "
				"selection"));
	connect(pasteAct, SIGNAL(triggered()), textEdit, SLOT(paste()));

	aboutAct = new QAction(tr("&Acerca de la aplicación"), this);
	aboutAct->setStatusTip(tr("Sobre la aplicación"));
	connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

	algorythmsHelpAct = new QAction(tr("&Sobre los algoritmos"),this);
	algorythmsHelpAct->setStatusTip(tr("Ayuda sobre los algoritmos"));
	connect(algorythmsHelpAct, SIGNAL(triggered()), this, SLOT(showAlgorythmsHelp()));

	aboutQtAct = new QAction(tr("Sobre &Qt"), this);
	aboutQtAct->setStatusTip(tr("Información sobre Qt"));
	connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));

	copyAct->setEnabled(false);
	connect(textEdit, SIGNAL(copyAvailable(bool)), cutAct,
			SLOT(setEnabled(bool)));
	connect(textEdit, SIGNAL(copyAvailable(bool)), copyAct,
			SLOT(setEnabled(bool)));
}

void MainWindow::createMenus() {
	fileMenu = menuBar()->addMenu(tr("&Fichero"));
	fileMenu->addAction(openAct);
	fileMenu->addAction(saveAct);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAct);

	menuBar()->addSeparator();

	helpMenu = menuBar()->addMenu(tr("&Información"));
	helpMenu->addAction(aboutAct);
	helpMenu->addAction(aboutQtAct);
	helpMenu->addAction(algorythmsHelpAct);
}

void MainWindow::createToolBars() {

	fileToolBar = new QToolBar();
	fileToolBar->setFloatable(false);
	fileToolBar->setOrientation(Qt::Vertical);
	fileToolBar->addAction(openAct);
	fileToolBar->addAction(saveAct);
	fileToolBar->addAction(executeAct);
	fileToolBar->setIconSize(QSize(300, 300));
}

void MainWindow::readSettings() {
	QSettings settings("Trolltech", "Application Example");
	QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
	QSize size = settings.value("size", QSize(400, 400)).toSize();
	resize(size);
	move(pos);
}

bool MainWindow::saveFile(const QString &fileName) {
	return true;
}

void MainWindow::setCurrentFile(const QString &fileName) {
	setWindowFilePath("FPG - José Ángel Rey Liñares");
}

QString MainWindow::strippedName(const QString &fullFileName) {
}

void MainWindow::execute() {
	Controlador *controlador = Controlador::Instance();
	Constants *constants = Constants::Instance();

	//1 - Modo imagen simple
	if (controlador->getApplicationMode() == constants->getSimpleImageMode()) {
		//Se obtiene el panel
		QWidget *widget =
				this->barraLateral->listaPaneles->pagesWidget->currentWidget();
		panel1 *panelImagen = dynamic_cast<panel1 *> (widget);
		if(panelImagen != 0){
			printf("El panel era de la clase 1");
		}else{
			printf("El panel no era de la clase 1, castéala a otra clase.");
		}

		if (!panelImagen->getMatrixListOrigin()->isEmpty()) {
			bool histeresis = panelImagen->histeresisCheck->isChecked();
			int radioGauss = panelImagen->radioGauss->value();
			double sigmaGauss = panelImagen->sigmaGauss->value();
			double lowerThreshold = panelImagen->umbralInferior->value();
			double higherThreshold = panelImagen->umbralSuperior->value();

			//Se obtiene el listado de imágenes del panel y se setea en el controlador de transferencia.
			controlador->fromQlist(panelImagen->getMatrixListOrigin());

			ImageFilterFactory fac =
					(ImageFilterFactory) *new ImageFilterFactory;
			ImageFilter* imageFilter = fac.getImageFilterInstance(
					panelImagen->algorythmSelect->currentText(), histeresis,
					radioGauss, sigmaGauss, lowerThreshold, higherThreshold);

			//Realizar la llamada al business de ejecución
			//Se instancia el business de negocio
			ImageProcessingBusiness imageProcessingBusiness;
			//Se realiza la llamada pasando el filtro (negocio determinará el tipo de filtro con la información disponible en el controlador.)
			imageProcessingBusiness.doProcess(*imageFilter);

			//panelImagen->loadResultImage(controlador->getResultImage());
			panelImagen->loadResultData(controlador->getMatrixListDestiny());
			//Despues del load realizamos el borrado:
			controlador->cleanResultList();
			//Habilitar o deshabilitar el botón de información de tiempos de ejecución
			this->enableInfoButton();
			//Habilitar o deshabilitar el botón de guardado de imágen/es
			this->enableSaveButton();
		} else {
			//Se generaría el popup de aviso al usuario.
			QMessageBox::warning(
					this,
					tr("¡Aviso!"),
					tr(
							"Debe seleccionar una imagen para poder realizar la ejecución de algún algoritmo."));
		}

		//Modo selección de múltiples imágenes.
	} else if (controlador->getApplicationMode()
			== constants->getMultipleImageMode()) {

		//Se obtiene el panel
		QWidget *widget =
				this->barraLateral->listaPaneles->pagesWidget->currentWidget();
		Panel2 *panelImagen = dynamic_cast<Panel2 *> (widget);
		if (!panelImagen->getMatrixListOrigin()->isEmpty()) {

			controlador->fromQlist(panelImagen->getMatrixListOrigin());

			bool histeresis = panelImagen->histeresisCheck->isChecked();
			int radioGauss = panelImagen->radioGauss->value();
			float sigmaGauss = panelImagen->sigmaGauss->value();
			float lowerThreshold = panelImagen->umbralInferior->value();
			float higherThreshold = panelImagen->umbralSuperior->value();

			controlador->fromQlist(panelImagen->getMatrixListOrigin());

			ImageFilterFactory fac =
					(ImageFilterFactory) *new ImageFilterFactory;
			ImageFilter* imageFilter = fac.getImageFilterInstance(
					panelImagen->algorythmSelect->currentText(), histeresis,
					radioGauss, sigmaGauss, lowerThreshold, higherThreshold);
			//Realizar la llamada al business de ejecución
			//Se instancia el business de negocio
			ImageProcessingBusiness imageProcessingBusiness;
			//Se realiza la llamada pasando el filtro (negocio determinará el tipo de filtro con la información disponible en el controlador.)
			imageProcessingBusiness.doProcess(*imageFilter);

			//Plasmar los resultados en el panel correcto según el modo de funcionamiento de la aplicación

			//panelImagen->loadResultImage(controlador->getResultImage());
			panelImagen->loadResultData(controlador->getMatrixListDestiny());
			//Despues del load realizamos el borrado:
			controlador->cleanResultList();
			//Habilitar o deshabilitar el botón de información de tiempos de ejecución
			this->enableInfoButton();
			//Habilitar o deshabilitar el botón de guardado de imágen/es
			this->enableSaveButton();
		} else {
			//Se generaría el popup de aviso al usuario.
			QMessageBox::warning(
					this,
					tr("¡Aviso!"),
					tr(
							"Debe seleccionar una imagen para poder realizar la ejecución de algún algoritmo."));
		}
		//Modo selección de múltiples algoritmos.
	} else if (controlador->getApplicationMode()
			== constants->getMultipleAlgorythmMode()) {

			QWidget
					*widget =
							this->barraLateral->listaPaneles->pagesWidget->currentWidget();
			Panel3 *panelImagen = dynamic_cast<Panel3 *> (widget);
			QList<ImageFilter*> *filters = panelImagen->getFilterList();
			if (!panelImagen->getMatrixListOrigin()->isEmpty()) {
			if(!filters->isEmpty()){
			controlador->fromQlist(panelImagen->getMatrixListOrigin());
			//Se instancia el business de negocio
			ImageProcessingBusiness imageProcessingBusiness;
			//Se realiza la llamada pasando el filtro (negocio determinará el tipo de filtro con la información disponible en el controlador.)
			imageProcessingBusiness.doProcess(filters);
			panelImagen->getMatrixListResult()->clear();
			//panelImagen->loadResultImage(controlador->getResultImage());
			panelImagen->loadResultData(controlador->getMatrixListDestiny());
			//Despues del load realizamos el borrado:
			controlador->cleanResultList();
			//Habilitar o deshabilitar el botón de información de tiempos de ejecución
			this->enableInfoButton();
			//Habilitar o deshabilitar el botón de guardado de imágen/es
			this->enableSaveButton();
			//Despues del load realizamos el borrado:
			controlador->cleanResultList();
			}else{
				//Se generaría el popup de aviso al usuario.
							QMessageBox::warning(
									this,
									tr("¡Aviso!"),
									tr(
											"Debe añadir al menos un algoritmo a la tabla de algoritmos para proceder a la ejecución de la funcionalidad (la imagen será procesada por todos los algoritmos presentes en la tabla, de forma separada y produciendo un resultado diferente por cada uno de los mismos)."));
			}
		} else {
			//Se generaría el popup de aviso al usuario.
			QMessageBox::warning(
					this,
					tr("¡Aviso!"),
					tr(
							"Debe seleccionar una imagen para poder realizar la ejecución de algún algoritmo."));
		}
		//modo gráfica 3D
	} else if (controlador->getApplicationMode() == constants->getPlotMode()) {
		//Se obtiene el panel
		QWidget *widget = this->barraLateral->listaPaneles->pagesWidget->currentWidget();
		Panel4 *panelImagen = dynamic_cast<Panel4 *> (widget);

		if (!panelImagen->getMatrixListOrigin()->isEmpty()) {
			panelImagen->getMatrixListResult()->clear();
			controlador->fromQlist(panelImagen->getMatrixListOrigin());

			bool histeresis = panelImagen->histeresisCheck->isChecked();
			int radioGauss = panelImagen->radioGauss->value();
			float sigmaGauss = panelImagen->sigmaGauss->value();
			float lowerThreshold = panelImagen->umbralInferior->value();
			float higherThreshold = panelImagen->umbralSuperior->value();

			ImageFilterFactory fac =
					(ImageFilterFactory) *new ImageFilterFactory;
			ImageFilter* imageFilter = fac.getImageFilterInstance(
					panelImagen->algorythmSelect->currentText(), histeresis,
					radioGauss, sigmaGauss, lowerThreshold, higherThreshold);
			//Realizar la llamada al business de ejecución
			//Se instancia el business de negocio
			ImageProcessingBusiness imageProcessingBusiness;
			//Se realiza la llamada pasando el filtro (negocio determinará el tipo de filtro con la información disponible en el controlador.)
			imageProcessingBusiness.doProcess(*imageFilter);

			//Plasmar los resultados en el panel correcto según el modo de funcionamiento de la aplicación
			//panelImagen->loadResultImage(controlador->getResultImage());
			panelImagen->loadResultData(controlador->getMatrixListDestiny());
			//Despues del load realizamos el borrado:
			controlador->cleanResultList();
			//Habilitar o deshabilitar el botón de información de tiempos de ejecución
			this->enableInfoButton();
			//Habilitar o deshabilitar el botón de guardado de imágen/es
			this->enableSaveButton();
			//Despues del load realizamos el borrado:
			controlador->cleanResultList();
		} else {
			//Se generaría el popup de aviso al usuario.
			QMessageBox::warning(
					this,
					tr("¡Aviso!"),
					tr(
							"Debe seleccionar una imagen para poder realizar la ejecución de algún algoritmo."));
		}
	} else if (controlador->getApplicationMode()
			== constants->getFrameCaptureMode()) {

		webCamProcess();
	}
}

void MainWindow::setupButtons() {
	//Instanciación y setup visual del botón de ejecución
	this->executeButton = new QPushButton();
	this->executeButton->setAutoFillBackground(true);
	this->executeButton->setIcon(QIcon(":/images/run.png"));
	this->executeButton->setToolTip("Ejecutar");
	this->executeButton->setIconSize(QSize(48, 32));
	this->executeButton->setStyle(
			new ManhattanStyle(QApplication::style()->objectName()));
	this->executeButton->setStyleSheet(
			"QPushButton{background: transparent; border: none;}QPushButton::hover{background-color: white; border: solid; border-color: black;}");

	//Instanciación y setup visual del botón de apertura de imágenes.
	this->stopButton = new QPushButton();
	this->stopButton->setEnabled(false);
	this->stopButton->setAutoFillBackground(true);
	this->stopButton->setIcon(QIcon(":images/stop.png"));
	this->stopButton->setToolTip("Para la ejecución");
	this->stopButton->setIconSize(QSize(48, 32));
	this->stopButton->setStyle(
			new ManhattanStyle(QApplication::style()->objectName()));
	this->stopButton->setStyleSheet(
			"QPushButton{background: transparent; border: none;}QPushButton::hover{background-color: white; border: solid; border-color: black;}");
	connect(stopButton, SIGNAL(clicked()), this, SLOT(webcamStop()));

	//Instanciación y setup visual del botón de apertura de imágenes.
	this->openButton = new QPushButton();
	this->openButton->setAutoFillBackground(true);
	this->openButton->setIcon(QIcon(":images/fileopen.png"));
	this->openButton->setToolTip("Comment");
	this->openButton->setIconSize(QSize(48, 32));
	this->openButton->setStyle(
			new ManhattanStyle(QApplication::style()->objectName()));
	this->openButton->setStyleSheet(
			"QPushButton{background: transparent; border: none;}QPushButton::hover{background-color: white; border: solid; border-color: black;}");

	//Instanciación y setup visual del botón de ayuda en modo.
	this->helpButton = new QPushButton();
	this->helpButton->setAutoFillBackground(true);
	this->helpButton->setIcon(QIcon(":images/help.png"));
	this->helpButton->setToolTip("Ayuda");
	this->helpButton->setIconSize(QSize(48, 32));
	this->helpButton->setStyle(
			new ManhattanStyle(QApplication::style()->objectName()));
	this->helpButton->setStyleSheet(
			"QPushButton{background: transparent; border: none;}QPushButton::hover{background-color: white; border: solid; border-color: black;}");

	//Instanciación y setup visual del botón de información de tiempo de ejeución..
	this->infoButton = new QPushButton();
	this->infoButton->setAutoFillBackground(true);
	this->infoButton->setIcon(QIcon(":images/lstopwatch.png"));
	this->infoButton->setToolTip("Información de tiempos de ejecución");
	this->infoButton->setEnabled(false);
	this->infoButton->setIconSize(QSize(48, 32));
	this->infoButton->setStyle(
			new ManhattanStyle(QApplication::style()->objectName()));
	this->infoButton->setStyleSheet(
			"QPushButton{background: transparent; border: none;}QPushButton::hover{background-color: white; border: solid; border-color: black;}");

	//Instanciación y setup visual del botón de guardado.
	this->saveButton = new QPushButton();
	this->saveButton->setAutoFillBackground(true);
	this->saveButton->setIcon(QIcon(":images/save.png"));
	this->saveButton->setToolTip("Guardado de imágen(es) procesadas.");
	this->saveButton->setEnabled(false);
	this->saveButton->setIconSize(QSize(48, 32));
	this->saveButton->setStyle(
			new ManhattanStyle(QApplication::style()->objectName()));
	this->saveButton->setStyleSheet(
			"QPushButton{background: transparent; border: none;}QPushButton::hover{background-color: white; border: solid; border-color: black;}");

	//Conexión de cada botón con su acción al realizar el pulsado.
	connect(executeButton, SIGNAL(pressed()), this, SLOT(execute()));
	connect(openButton, SIGNAL(pressed()), this, SLOT(open()));
	connect(helpButton, SIGNAL(pressed()), this, SLOT(showHelpMessage()));
	connect(infoButton, SIGNAL(pressed()), this,
			SLOT(showInfoExecutionMessage()));
	connect(saveButton, SIGNAL(pressed()), this, SLOT(save()));

}

void MainWindow::setupTabBar() {
	//Se instancia la barra como widget principal de la pantalla
	barraLateral = new FancyTabWidget(this);
	//Se declaran las entradas para el mismo (una para cada modo)
	//1- Modo ejecución simple
	barraLateral->insertTab(0, new QWidget(this), QIcon(":images/image.png"),
			"Ejec. Simple");
	//2- Modo ejecución múltiple algoritmo (simple imagen)
	barraLateral->insertTab(1, new QWidget(this), QIcon(":images/images.png"),
			"Múlt. Imag.");
	//3- Modo ejecución múltiple imagen (simple algoritmo)
	barraLateral->insertTab(2, new QWidget(this), QIcon(":images/grid.png"),
			"Múlt. Alg.");
	//4- Modo gráfica 3D
	barraLateral->insertTab(3, new QWidget(this), QIcon(":images/graph.png"),
			"Gráfica 3D");
	//5- Modo cámara
	barraLateral->insertTab(4, new QWidget(this), QIcon(":images/camera.png"),
			"Webcam");
	//6- Información de la tarjeta gráfica
	barraLateral->insertTab(5, new QWidget(this), QIcon(":images/nvidia.png"),
			"Info.");
	//7- Configuración (selección de tarjeta y modo de ejecución)
	barraLateral->insertTab(6, new QWidget(this), QIcon(":images/setting.png"),
			"Config.");
	//Se les asigna una posición en el menú
	barraLateral->setTabEnabled(0, true);
	barraLateral->setTabEnabled(1, true);
	barraLateral->setTabEnabled(2, true);
	barraLateral->setTabEnabled(3, true);
	barraLateral->setTabEnabled(4, true);
	barraLateral->setTabEnabled(5, true);
	barraLateral->setTabEnabled(6, true);
}

void MainWindow::setupInferiorMenu() {
	//Se asignan los botones a la zona inferior de la barra lateral
	barraLateral->insertCornerWidget(1, executeButton);
	barraLateral->insertCornerWidget(2, stopButton);
	barraLateral->insertCornerWidget(3, openButton);
	barraLateral->insertCornerWidget(4, saveButton);
	barraLateral->insertCornerWidget(5, helpButton);
	barraLateral->insertCornerWidget(6, infoButton);
}

void MainWindow::enableInfoButton() {
	int sizeValue = 0;
	//Se obtiene el modo actual de funcionamiento de la aplicación
	QString mode = Controlador::Instance()->getApplicationMode();
	//Se castea a la clase genérica
	//En función del modo de funcionamiento actual obtenemos la cola de mensajes de tiempo de los algoritmos
	//correspondiente y miramos su longitud, si tiene mensajes habilitamos el botón en caso contrario lo deshabilitamos
	if (mode == Constants::Instance()->getSimpleImageMode()) {
		sizeValue = Controlador::Instance()->getSimpleImageExecution()->size();
	} else if (mode == Constants::Instance()->getMultipleImageMode()) {
		sizeValue
				= Controlador::Instance()->getMultipleImageExecution()->size();
	} else if (mode == Constants::Instance()->getMultipleAlgorythmMode()) {
		sizeValue
				= Controlador::Instance()->getMultipleAlgorythmExecution()->size();
	}

	if (sizeValue > 0) {
		this->infoButton->setEnabled(true);
	} else {
		this->infoButton->setEnabled(false);
	}
}

void MainWindow::enableSaveButton() {
	int sizeValue = 0;
	//Se obtiene el panel actualseleccionado
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();

	//Se obtiene el listado de imágenes resultado
	PanelBase *panelImagen = dynamic_cast<PanelBase *> (widget);

	sizeValue = panelImagen->getMatrixListResult()->size();

	if (sizeValue > 0) {
		this->saveButton->setEnabled(true);
	} else {
		this->saveButton->setEnabled(false);
	}
}
void MainWindow::showHelpMessage() {
	//Se obtiene el panel actual y se pinta su mensaje de ayuda en un dialog.
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();
	PanelBase *panel = dynamic_cast<PanelBase *> (widget);

	QMessageBox::about(this, tr("Ayuda"), panel->getHelpData());

}

void MainWindow::showInfoExecutionMessage() {
	//Se obtiene el panel actual y se pinta su mensaje de ayuda en un dialog.
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();
	PanelBase *panel = dynamic_cast<PanelBase *> (widget);

	//QMessageBox::about(this, tr("Tiempos de ejecución"),
	//			panel->getExecutionData());
	TimeDialog *dialog = new TimeDialog;
	dialog->show();

}

void MainWindow::webCamProcess() {
	Controlador *controlador = Controlador::Instance();
	if(controlador->getIsGpuMode() == 2){
		//Mostrar popup de aviso de que elmodo webcam no está destinado a comparación, no soporta
		//el modo CPU vs GPU.
		//Se generaría el popup de aviso al usuario.
					QMessageBox::warning(
							this,
							tr("¡Aviso!"),
							tr(
									"El modo de procesamiento de cámara web no está destinado a comparación y no admite el modo de ejecución CPU vs GPU."
									"Por favor, cambie el modo de la aplicación a ejecución CPU o ejecución GPU para emplear esta funcionalidad, gracias."));
	}else{
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();
	Panel6 *panelImagen = dynamic_cast<Panel6 *> (widget);

	int fd = -1;
	char *dev_name;
	dev_name = "/dev/video0";

	fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);

	if (fd < 0) {
		this->eWriteLine(
				"No se han encontrado dispositivos compatibles con v4l2 en /dev/video0 por favor revise la configuración de su equipo para poder utilizar esta funcionalidad.");
		v4l2_close(fd);
	} else if ((fd >= 0) && (panelImagen->t->devam == false)) {
		v4l2_close(fd);
		panelImagen->t->devam = true;
		panelImagen->t->start();
	}
	}
}

void MainWindow::webcamStop() {
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();
	Panel6 *panelImagen = dynamic_cast<Panel6 *> (widget);

	if(panelImagen->t->devam == true){
		panelImagen->t->devam = false;
		panelImagen->setCanSave(true);
		this->saveButton->setEnabled(true);
	}

}

void MainWindow::panelChanged() {
	this->manageSidebarButtons();
}

void MainWindow::manageSidebarButtons() {
	QWidget *widget =
			this->barraLateral->listaPaneles->pagesWidget->currentWidget();
	PanelBase *panel = dynamic_cast<PanelBase *> (widget);

	//Se habilitan los botones según indique el panel.
	this->executeButton->setEnabled(panel->getRunButtonNeeded());
	this->openButton->setEnabled(panel->getOpenButtonNeeded());
	this->stopButton->setEnabled(panel->getStopButtonNeeded());

	if (panel->getTimeButtonNeeded()) {
		this->enableInfoButton();
	}else{
		this->infoButton->setEnabled(false);
	}

	if (panel->getSaveButtonNeeded()) {
		this->enableSaveButton();
	}else{
		this->saveButton->setEnabled(false);
	}
}

void MainWindow::eWriteLine(QString message) {

	QMessageBox::critical(this, tr("Error"), message);

}

void MainWindow::showAlgorythmsHelp() {
	QMessageBox::about(
			this,
			tr("Ayuda Sobre Algoritmos"),
			tr(
					"La aplicación permite la ejecución de un grupo de distintos algoritmos de detección de contornos sobre imágenes.<br/>"
					"Estos algoritmos se pueden clasificar en dos grupos, los algoritmos de primer y segundo orden, en base a los filtros empleados.<br/>"
					"Los <b>algoritmos de primer orden</b> son aquellos que emplean filtros de primera orden que permiten aproximar la primera derivada<br/>"
					"en cada punto de la imagen. Los algoritmos de <b>segundo orden</b> son aquellos que emplean filtros de segundo orden que permiten <br/>"
					"realizar la aproximación de la segunda derivada en cada punto de la imagen.<br/>"
					"A continuación se muestran los algoritmos ofrecidos por la aplicación clasificados en sus grupos:<br/><br/>"
					"<b>Algoritmos de primer orden:</b><br/>"
					"- Algoritmo de Sobel.<br/>"
					"- Algoritmo de Sobel Square.<br/>"
					"- Algoritmo de Robert Cross.<br/>"
					"- Algoritmo de Prewitt.<br/>"
					"- Algoritmo de Canny.<br/><br/>"
					"<b>Algoritmos de segundo orden:</b><br/>"
					"- Algoritmo de Laplace.<br/>"
					"- Algoritmo de Laplace de Gauss.<br/><br/>"
					"Una descripción más detallada de cada uno de los algoritmos y de sus parámetros, en caso de proceder, puede ser vista pulsando<br/>"
					"el icono de ayuda a la derecha del desplegable de selección de algoritmos en cualquier ventana que permita la ejecución de algoritmos.<br/>"
					"Dicha ayuda mostrará la descripción del algoritmo seleccionado en ese momento en el desplegable."));

}
