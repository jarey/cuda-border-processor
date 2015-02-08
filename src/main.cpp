/*
 * main.cpp
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: clase principal de invocación de la aplicación.
 */

#include <QApplication>

#include "./src/ui/mainwindow.h"
#include <qfile.h>
#include <QSplashScreen>
#include <time.h>
#include <QTextCodec>
#include <QApplication>
#include <QMessageBox>
#include <QDebug>

#include "./src/cudainfo/log.h"
#include "./src/cudainfo/czdialog.h"
#include "./src/cudainfo/cudainfo.h"
#include "./src/cudainfo/version.h"
#include "./src/common/Controlador.h"


/*
 * Función: testCudaPresent
 * Propósito: Realiza el chequeo de si existen dispositivos CUDA en el equipo.
 * 			  Realiza la invocación al método CZCudaCheck definida en cudainfo.h.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
bool testCudaPresent() {
	bool res = CZCudaCheck();
	return res;
}

/*
 * Función: getCudaDeviceNum
 * Propósito: Devuelve el número de dispositivos CUDA encontrados en el equipo.
 * 			  Realiza la llamada al método CZCudaDeviceFound definido en cudainfo.h
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
int getCudaDeviceNum() {
	int res = CZCudaDeviceFound();
	return res;
}



/*
 * Función: main
 * Propósito: Realiza la configuración del programa y lo arranca.
 * Creado: 25/03/2011
 * Autor: jose
 *
 */
int main(int argc, char *argv[])
{
	//inicializamos los recursos (path de imágenes, etc de la aplicación).
    Q_INIT_RESOURCE(application);

    //Aplicamos la configuración de caracteres UTF a la aplciación
    QTextCodec *linuxCodec = QTextCodec::codecForName("UTF-8");
    QTextCodec::setCodecForTr(linuxCodec);
    QTextCodec::setCodecForCStrings(linuxCodec);
    QTextCodec::setCodecForLocale(linuxCodec);

    //Instanciamos el programa principal.
    QApplication app(argc, argv);
    //Configuramos el nombre de la organización y de la aplicación.
    app.setOrganizationName("PFG");
    app.setApplicationName("PFG");

    //Sacamos la línea de log que indica que se comienza el scaneo de dispositivos cuda.
	CZLog(CZLogLevelHigh, "Se inicializa el escaneo de dispositivos CUDA...");


	//Se crea la pantalla splash a apartir de la imagen en el fichero de recursos.
	QPixmap pixmap(":/images/Border Detector SplashScreen.png");
	splash = new CZSplashScreen(pixmap, 2);
	splash->show();
	//Se muestra un mensaje informativo en la pantalla splash, informando de que se está
	//realizando la comprobación de dispositivos cuda.
	splash->showMessage(QObject::tr("Comprobando tarjetas compatibles con CUDA ..."),
		Qt::AlignLeft | Qt::AlignBottom);
	app.processEvents();

	//Si la función que devuelve si existe algún dispositivo cuda en el sistema devuelve
	//falso se muestra una ventana emergente para indicar de este suceso.
	if(!testCudaPresent()) {
		QMessageBox::critical(0, QObject::tr(CZ_NAME_LONG),
			QObject::tr("No se han encontrado dispositivos compatibles con CUDA.!"));
	}

	//Si el número de dispositivos CUDA encontrados en el sistema es 0, entonces se levanta una ventana
	//emergente para indicar al usuario que solo podrá realizar la ejecución de las funcionalidades
	//de la aplicación en modo CPU.
	int devs = getCudaDeviceNum();
	if(devs == 0) {
		Controlador::Instance()->setIsCudaCapable(0);
		QMessageBox::critical(0, QObject::tr(CZ_NAME_LONG),
			QObject::tr("¡No se han encontrado dispositivos compatibles con CUDA.! Sólo podrá ejecutar funcionalidad relativa a CPU"));
	}else{
		//En caso de que se encuentren dispositivos CUDA se muestra un mensaje en la pantalla
		//splash de carga indicando el número de dispositivos encontrados.
		Controlador::Instance()->setIsCudaCapable(1);
	splash->showMessage(QObject::tr("Se ha encontrado %1 dispositivo(s) CUDA ...").arg(devs),
		Qt::AlignLeft | Qt::AlignBottom);
	app.processEvents();
	}

	//Se crea y se muestra la ventana principal del programa.
	MainWindow mainWin;
	mainWin.show();
	//Se elimina la ventana splash de carga.
	splash->finish(&mainWin);

	return app.exec();
}
