/*
 * mainwindow.h
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: ventana principal de la aplicación. Realiza la gestión de las invocaciones a negocio.
 *  La carga de imágenes, el guardado, la habilitación y deshabilitación de botones y la gestión
 *  de los paneles.
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <qpushbutton.h>
#include "panelList.h"
#include "fancytabwidget.h"
#include "./src/common/Controlador.h"
#include "./src/canny/CImage.h"
#include "./src/ui/fancytabwidget.h"

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
class QPlainTextEdit;
QT_END_NAMESPACE

using namespace Core;
using namespace Core::Internal;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();

protected:
    void closeEvent(QCloseEvent *event);

private slots:
    void open();
    bool save();
    void about();
    void execute();
    void showHelpMessage();
    void showInfoExecutionMessage();
    void panelChanged();
    void webcamStop();
    void showAlgorythmsHelp();

private:
    //Funciones públicas
    void createActions();
    void createMenus();
    void createToolBars();
    void readSettings();
    void setupButtons();
    void setupTabBar();
    void setupInferiorMenu();
    bool saveFile(const QString &fileName);
    void setCurrentFile(const QString &fileName);
    QString strippedName(const QString &fullFileName);
    void enableInfoButton();
    void enableSaveButton();
    void webCamProcess();

    void manageSidebarButtons();

    //Componentes publicos
    QPlainTextEdit *textEdit;
    QString curFile;
    QMenu *fileMenu;
    QMenu *editMenu;
    QMenu *helpMenu;
    QToolBar *fileToolBar;
    QToolBar *editToolBar;
    QAction *newAct;
    QAction *openAct;
    QAction *saveAct;
    QAction *saveAsAct;
    QAction *exitAct;
    QAction *cutAct;
    QAction *copyAct;
    QAction *pasteAct;
    QAction *aboutAct;
    QAction *aboutQtAct;
    QAction *executeAct;
    QAction *algorythmsHelpAct;
    FancyTabWidget *barraLateral;
    Controlador *controlador;
    //Se añaden los botones para el menú inferior de ejecución rápida.
    QPushButton *executeButton;
    QPushButton *stopButton;
    QPushButton *openButton;
    QPushButton *helpButton;
    QPushButton *infoButton;
    QPushButton *saveButton;

public slots:
	void eWriteLine(QString message);

};

#endif
