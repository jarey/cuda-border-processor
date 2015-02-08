/*
 * czdialog.cpp
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: panel de mostrado de la información de la GPU. Esta clase reutiliza código
 *  obtenido del proyecto open-source CUDA Z, adaptando dicho código para integrarlo en el proyecto,
 *  haciendo uso de la licencia GPL.
 */

#include <QMenu>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QDebug>
#include <time.h>
#include "log.h"
#include "czdialog.h"
#include "version.h"
#include "./src/common/Constants.h"

#define CZ_TIMER_REFRESH	2000	/*!< Test results update timer period (ms). */

/*!
	\def CZ_OS_PLATFORM_STR Platform ID string.
*/
#if defined(Q_OS_WIN)
#define CZ_OS_PLATFORM_STR	"win32"
#elif defined(Q_OS_MAC)
#define CZ_OS_PLATFORM_STR	"macosx"
#elif defined(Q_OS_LINUX)
#define CZ_OS_PLATFORM_STR	"linux"
#else
#error Your platform is not supported by CUDA! Or it does but I know nothing about this...
#endif

/*!
	\class CZSplashScreen
	\brief Splash screen with multiline logging effect.
*/

/*!
	\brief Creates a new #CZSplashScreen and initializes internal
	parameters of the class.
*/
CZSplashScreen::CZSplashScreen(
	const QPixmap &pixmap,	/*!< Picture for window background. */
	int maxLines,		/*!< Number of lines in boot log. */
	Qt::WindowFlags f	/*!< Window flags. */
):	QSplashScreen(pixmap, f),
	m_maxLines(maxLines) {
	m_message = QString::null;
	m_lines = 0;
	m_alignment = Qt::AlignLeft;
	m_color = Qt::black;
}

/*!
	\brief Creates a new #CZSplashScreen with the given \a parent and
	initializes internal parameters of the class.
*/
CZSplashScreen::CZSplashScreen(
	QWidget *parent,	/*!< Parent of widget. */
	const QPixmap &pixmap,	/*!< Picture for window background. */
	int maxLines,		/*!< Number of lines in boot log. */
	Qt::WindowFlags f	/*!< Window flags. */
):	QSplashScreen(parent, pixmap, f),
	m_maxLines(maxLines) {
	m_message = QString::null;
	m_lines = 0;
	m_alignment = Qt::AlignLeft;
	m_color = Qt::black;
}

/*!
	\brief Class destructor.
*/
CZSplashScreen::~CZSplashScreen() {
}

/*!
	\brief Sets the maximal number of lines in log.
*/
void CZSplashScreen::setMaxLines(
	int maxLines		/*!< Number of lines in log. */
) {
	if(maxLines >= 1) {
		m_maxLines = maxLines;
		if(m_lines > m_maxLines) {
			deleteTop(m_lines - m_maxLines);
			QSplashScreen::showMessage(m_message, m_alignment, m_color);
		}
	}
}

/*!
	\brief Returns the maximal number of lines in log.
	\return number of lines in log.
*/
int CZSplashScreen::maxLines() {
	return m_maxLines;
}

/*!
	\brief Adds new message line in log.
*/
void CZSplashScreen::showMessage(
	const QString &message,	/*!< Message text. */
	int alignment,		/*!< Placement of log in window. */
	const QColor &color	/*!< Color used for protocol display. */
) {

	m_alignment = alignment;
	m_color = color;

	if(m_message.size() != 0) {
		m_message += '\n' + message;
	} else {
		m_message = message;
	}
	QStringList linesList = m_message.split('\n');
	m_lines = linesList.size();

	if(m_lines > m_maxLines) {
		deleteTop(m_lines - m_maxLines);
	}

	QSplashScreen::showMessage(m_message, m_alignment, m_color);
}

/*!
	\brief Removes all messages being displayed in log.
*/
void CZSplashScreen::clearMessage() {
	m_message = QString::null;
	m_lines = 0;
	QSplashScreen::showMessage(m_message, m_alignment, m_color);
}

/*!
	\brief Removes first \a lines entries in log.
*/
void CZSplashScreen::deleteTop(
	int lines		/*!< Number of lines to be removed. */
) {
	QStringList linesList = m_message.split('\n');
	for(int i = 0; i < lines; i++) {
		linesList.removeFirst();
	}

	m_message = linesList.join(QString('\n'));
	m_lines -= lines;
}

/*!
	\brief Splash screen of application.
*/
CZSplashScreen *splash;

/*!
	\class CZDialog
	\brief This class implements main window of the application.
*/

/*!
	\brief Creates a new #CZDialog with the given \a parent.
	This function does following steps:
	- Sets up GUI.
	- Setup CUDA-device information containers and add them in list.
	- Sets up connections.
	- Fills up data in to tabs of GUI.
	- Starts Performance data update timer.
*/
CZDialog::CZDialog(){

	http = NULL;

	setupUi();
	retranslateUi();
	connect(comboDevice, SIGNAL(activated(int)), SLOT(slotShowDevice(int)));

	QMenu *exportMenu = new QMenu(pushExport);
	exportMenu->addAction(tr("to &Text"), this, SLOT(slotExportToText()));
	exportMenu->addAction(tr("to &HTML"), this, SLOT(slotExportToHTML()));
	pushExport->setMenu(exportMenu);
	readCudaDevices();
	setupDeviceList();
	qDebug()<< "El indice del combo a procesar " << comboDevice->currentIndex();
	if(comboDevice->currentIndex()>=0){
	setupDeviceInfo(comboDevice->currentIndex());
	setupAboutTab();
	updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), SLOT(slotUpdateTimer()));
	updateTimer->start(CZ_TIMER_REFRESH);
	}
	labelAppUpdate->setText(tr("Looking for new version..."));

	//Se setean los valores de control de habilitación de los botones de acciones en pantalla.
	setRunButtonNeeded(false);
	setStopButtonNeeded(false);
	setTimeButtonNeeded(false);
	setOpenButtonNeeded(false);
	setSaveButtonNeeded(false);
}

/*!
	\brief Class destructor.
	This function makes class data cleanup actions.
*/
CZDialog::~CZDialog() {
	updateTimer->stop();
	delete updateTimer;
	freeCudaDevices();
}

/*!
	\brief Reads CUDA devices information.
	For each of detected CUDA-devices does following:
	- Initialize CUDA-data structure.
	- Reads CUDA-information about device.
	- Shows progress message in splash screen.
	- Starts Performance calculation procedure.
	- Appends entry in to device-list.
*/
void CZDialog::readCudaDevices() {
	int num = getCudaDeviceNumber();
	for(int i = 0; i < num; i++) {
		CZCudaDeviceInfo *info = new CZCudaDeviceInfo(i);
		if(info->info().major != 0) {
			info->waitPerformance();
			connect(info, SIGNAL(testedPerformance(int)), SLOT(slotUpdatePerformance(int)));
			deviceList.append(info);
		}
	}
}

/*!
	\brief Cleans up after bandwidth tests.
*/
void CZDialog::freeCudaDevices() {

	while(deviceList.size() > 0) {
		CZCudaDeviceInfo *info = deviceList[0];
		deviceList.removeFirst();
		delete info;
	}
}

/*!
	\brief Gets number of CUDA devices.
	\return number of CUDA-devices in case of success, \a 0 if no CUDA-devies were found.
*/
int CZDialog::getCudaDeviceNumber() {
	return CZCudaDeviceFound();
}

/*!
	\brief Puts devices in combo box.
*/
void CZDialog::setupDeviceList() {
	comboDevice->clear();

	for(int i = 0; i < deviceList.size(); i++) {
		comboDevice->addItem(QString("%1: %2").arg(i).arg(deviceList[i]->info().deviceName));
	}
}

/*!
	\brief This slot shows in dialog information about given device.
*/
void CZDialog::slotShowDevice(
	int index			/*!< Index of device in list. */
) {
	setupDeviceInfo(index);
	if(checkUpdateResults->checkState() == Qt::Checked) {
		CZLog(CZLogLevelMid, "Switch device -> update performance for device %d", index);
		deviceList[index]->testPerformance(index);
	}
}

/*!
	\brief This slot updates performance information of device
	pointed by \a index.
*/
void CZDialog::slotUpdatePerformance(
	int index			/*!< Index of device in list. */
) {
	if(index == comboDevice->currentIndex())
	setupPerformanceTab(deviceList[index]->info());
}

/*!
	\brief This slot updates performance information of current device
	every timer tick.
*/
void CZDialog::slotUpdateTimer() {

	int index = comboDevice->currentIndex();
	if(checkUpdateResults->checkState() == Qt::Checked) {
		CZLog(CZLogLevelMid, "Timer shot -> update performance for device %d", index);
		deviceList[index]->testPerformance(index);
	} else {
		CZLog(CZLogLevelMid, "Timer shot -> update ignored");
	}
}

/*!
	\brief Places in dialog's tabs information about given device.
*/
void CZDialog::setupDeviceInfo(
	int dev				/*!< Number of CUDA-device. */
) {
	setupCoreTab(deviceList[dev]->info());
	setupMemoryTab(deviceList[dev]->info());
	setupPerformanceTab(deviceList[dev]->info());
}

/*!
	\brief Fill tab "Core" with CUDA devices information.
*/
void CZDialog::setupCoreTab(
	struct CZDeviceInfo &info	/*!< Information about CUDA-device. */
) {
	QString deviceName(info.deviceName);

	labelNameText->setText(deviceName);
	labelCapabilityText->setText(QString("%1.%2").arg(info.major).arg(info.minor));
	labelClockText->setText(QString("%1 %2").arg((double)info.core.clockRate / 1000).arg(tr("MHz")));
	if(info.core.muliProcCount == 0)
		labelMultiProcText->setText("<i>" + tr("Unknown") + "</i>");
	else
		labelMultiProcText->setNum(info.core.muliProcCount);
	labelWarpText->setNum(info.core.SIMDWidth);
	labelRegsText->setNum(info.core.regsPerBlock);
	labelThreadsText->setNum(info.core.maxThreadsPerBlock);
	if(info.core.watchdogEnabled == -1)
		labelWatchdogText->setText("<i>" + tr("Desconocio") + "</i>");
	else
		labelWatchdogText->setText(info.core.watchdogEnabled? tr("Sí"): tr("No"));
	labelThreadsDimTextX->setNum(info.core.maxThreadsDim[0]);
	labelThreadsDimTextY->setNum(info.core.maxThreadsDim[1]);
	labelThreadsDimTextZ->setNum(info.core.maxThreadsDim[2]);
	labelGridDimTextX->setNum(info.core.maxGridSize[0]);
	labelGridDimTextY->setNum(info.core.maxGridSize[1]);
	labelGridDimTextZ->setNum(info.core.maxGridSize[2]);

	labelDeviceLogo->setPixmap(QPixmap(":src/cudaz/img/logo-unknown.png"));
	if(deviceName.contains("tesla", Qt::CaseInsensitive)) {
		labelDeviceLogo->setPixmap(QPixmap(":src/cudaz/img/logo-tesla.png"));
	} else
	if(deviceName.contains("quadro", Qt::CaseInsensitive)) {
		labelDeviceLogo->setPixmap(QPixmap(":src/cudaz/img/logo-quadro.png"));
	} else
	if(deviceName.contains("geforce", Qt::CaseInsensitive)) {
		labelDeviceLogo->setPixmap(QPixmap(":src/cudaz/img/logo-geforce.png"));
	}
}

/*!
	\brief Fill tab "Memory" with CUDA devices information.
*/
void CZDialog::setupMemoryTab(
	struct CZDeviceInfo &info	/*!< Information about CUDA-device. */
) {
	labelTotalGlobalText->setText(QString("%1 %2")
		.arg((double)info.mem.totalGlobal / (1024 * 1024)).arg(tr("MB")));
	labelSharedText->setText(QString("%1 %2")
		.arg((double)info.mem.sharedPerBlock / 1024).arg(tr("KB")));
	labelPitchText->setText(QString("%1 %2")
		.arg((double)info.mem.maxPitch / 1024).arg(tr("KB")));
	labelTotalConstText->setText(QString("%1 %2")
		.arg((double)info.mem.totalConst / 1024).arg(tr("KB")));
	labelTextureAlignmentText->setNum(info.mem.textureAlignment);
	labelGpuOverlapText->setText(info.mem.gpuOverlap? tr("Yes"): tr("No"));
}

/*!
	\brief Fill tab "Performance" with CUDA devices information.
*/
void CZDialog::setupPerformanceTab(
	struct CZDeviceInfo &info	/*!< Information about CUDA-device. */
) {

	if(info.band.copyHDPin == 0)
		labelHDRatePinText->setText("--");
	else
		labelHDRatePinText->setText(QString("%1 %2")
			.arg((double)info.band.copyHDPin / 1024).arg(tr("MB/s")));

	if(info.band.copyHDPage == 0)
		labelHDRatePageText->setText("--");
	else
		labelHDRatePageText->setText(QString("%1 %2")
			.arg((double)info.band.copyHDPage / 1024).arg(tr("MB/s")));

	if(info.band.copyDHPin == 0)
		labelDHRatePinText->setText("--");
	else
		labelDHRatePinText->setText(QString("%1 %2")
			.arg((double)info.band.copyDHPin / 1024).arg(tr("MB/s")));

	if(info.band.copyDHPage == 0)
		labelDHRatePageText->setText("--");
	else
		labelDHRatePageText->setText(QString("%1 %2")
			.arg((double)info.band.copyDHPage / 1024).arg(tr("MB/s")));

	if(info.band.copyDD == 0)
		labelDDRateText->setText("--");
	else
		labelDDRateText->setText(QString("%1 %2")
			.arg((double)info.band.copyDD / 1024).arg(tr("MB/s")));

	if(info.perf.calcFloat == 0)
		labelFloatRateText->setText("--");
	else
		labelFloatRateText->setText(QString("%1 %2")
			.arg((double)info.perf.calcFloat / 1000).arg(tr("Mflop/s")));

	if(((info.major > 1)) ||
		((info.major == 1) && (info.minor >= 3))) {
		if(info.perf.calcDouble == 0)
			labelDoubleRateText->setText("--");
		else
			labelDoubleRateText->setText(QString("%1 %2")
				.arg((double)info.perf.calcDouble / 1000).arg(tr("Mflop/s")));
	} else {
		labelDoubleRateText->setText("<i>" + tr("No soportado") + "</i>");
	}

	if(info.perf.calcInteger32 == 0)
		labelInt32RateText->setText("--");
	else
		labelInt32RateText->setText(QString("%1 %2")
			.arg((double)info.perf.calcInteger32 / 1000).arg(tr("Miop/s")));

	if(info.perf.calcInteger24 == 0)
		labelInt24RateText->setText("--");
	else
		labelInt24RateText->setText(QString("%1 %2")
			.arg((double)info.perf.calcInteger24 / 1000).arg(tr("Miop/s")));
}

/*!
	\brief Fill tab "About" with information about this program.
*/
void CZDialog::setupAboutTab() {
}

/*!
	\fn CZDialog::getOSVersion
	\brief Get OS version string.
	\return string that describes version of OS we running at.
*/
#ifdef Q_OS_WIN
#include <windows.h>
typedef BOOL (WINAPI *IsWow64Process_t)(HANDLE, PBOOL);

QString CZDialog::getOSVersion() {
	QString OSVersion = "Windows";

	BOOL is_os64bit = FALSE;
	IsWow64Process_t p_IsWow64Process = (IsWow64Process_t)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "IsWow64Process");
	if(p_IsWow64Process != NULL) {
		if(!p_IsWow64Process(GetCurrentProcess(), &is_os64bit)) {
			is_os64bit = FALSE;
	        }
	}

	OSVersion += QString(" %1").arg(
		(is_os64bit == TRUE)? "AMD64": "x86");

/*	GetSystemInfo(&systemInfo);
	OSVersion += QString(" %1").arg(
		(systemInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64)? "AMD64":
		(systemInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_IA64)? "IA64":
		(systemInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL)? "x86":
		"Unknown architecture");*/

	OSVERSIONINFO versionInfo;
	ZeroMemory(&versionInfo, sizeof(OSVERSIONINFO));
	versionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&versionInfo);
	OSVersion += QString(" %1.%2.%3 %4")
		.arg(versionInfo.dwMajorVersion)
		.arg(versionInfo.dwMinorVersion)
		.arg(versionInfo.dwBuildNumber)
		.arg(QString::fromWCharArray(versionInfo.szCSDVersion));

	return OSVersion;
}
#elif defined (Q_OS_LINUX)
#include <QProcess>
QString CZDialog::getOSVersion() {
	QProcess uname; 

	uname.start("uname", QStringList() << "-srvm");
	if(!uname.waitForFinished())
		return QString("Linux (unknown)");
	QString OSVersion = uname.readLine();

	return OSVersion.remove('\n');
}
#else//!Q_WS_WIN
#error Function getOSVersion() is not implemented for your platform!
#endif//Q_WS_WIN

/*!
	\brief Export information to plane text file.
*/
void CZDialog::slotExportToText() {

	struct CZDeviceInfo info = deviceList[comboDevice->currentIndex()]->info();

	QString fileName = QFileDialog::getSaveFileName(this, tr("Save Text as..."),
		tr("%1.txt").arg(tr(CZ_NAME_SHORT)), tr("Text files (*.txt);;All files (*.*)"));

	if(fileName.isEmpty())
		return;

	CZLog(CZLogLevelMid, "Export to text as %s", fileName);

	QFile file(fileName);
	if(!file.open(QFile::WriteOnly | QFile::Text)) {
		QMessageBox::warning(this, tr(CZ_NAME_SHORT),
			tr("Cannot write file %1:\n%2.").arg(fileName).arg(file.errorString()));
		return;
	}

	QTextStream out(&file);
	QString title = tr("%1 Informe").arg(tr(CZ_NAME_SHORT));
	QString subtitle;

	out << title << endl;
	for(int i = 0; i < title.size(); i++)
		out << "=";
	out << endl;
	out << QString("%1: %2").arg(tr("Version")).arg(CZ_VERSION);
#ifdef CZ_VER_STATE
	out << QString(" %1 %2 %3 ").arg("Built").arg(CZ_DATE).arg(CZ_TIME);
#endif//CZ_VER_STATE
	out << endl;
	out << CZ_ORG_URL_MAINPAGE << endl;
	out << QString("%1: %2").arg(tr("OS Version")).arg(getOSVersion()) << endl;
	out << endl;

	subtitle = tr("Información sobre el núcleo");
	out << subtitle << endl;
	for(int i = 0; i < subtitle.size(); i++)
		out << "-";
	out << endl;
	out << "\t" << QString("%1: %2").arg(tr("Nombre")).arg(info.deviceName) << endl;
	out << "\t" << QString("%1: %2.%3").arg(tr("Capacidad computacional")).arg(info.major).arg(info.minor) << endl;
	out << "\t" << QString("%1: %2 %3").arg(tr("Velocidad de reloj")).arg((double)info.core.clockRate / 1000).arg(tr("MHz")) << endl;
	out << "\t" << tr("Multiprocessors") << ": ";
	if(info.core.muliProcCount == 0)
		out << tr("Unknown") << endl;
	else
		out << info.core.muliProcCount << endl;
	out << "\t" << QString("%1: %2").arg(tr("Warp Size")).arg(info.core.SIMDWidth) << endl;
	out << "\t" << QString("%1: %2").arg(tr("Regs Per Block")).arg(info.core.regsPerBlock) << endl;
	out << "\t" << QString("%1: %2").arg(tr("Hilos por bloque")).arg(info.core.maxThreadsPerBlock) << endl;
	out << "\t" << QString("%1: %2").arg(tr("Watchdog Enabled")).arg(info.core.watchdogEnabled? tr("Yes"): tr("No")) << endl;
	out << "\t" << QString("%1: %2 x %3 x %4").arg(tr("Dimensión de los hilos")).arg(info.core.maxThreadsDim[0]).arg(info.core.maxThreadsDim[1]).arg(info.core.maxThreadsDim[2]) << endl;
	out << "\t" << QString("%1: %2 x %3 x %4").arg(tr("Dimensión de lso grids")).arg(info.core.maxGridSize[0]).arg(info.core.maxGridSize[1]).arg(info.core.maxGridSize[2]) << endl;
	out << endl;

	subtitle = tr("Información de la memoria");
	out << subtitle << endl;
	for(int i = 0; i < subtitle.size(); i++)
		out << "-";
	out << endl;
	out << "\t" << QString("%1: %2 %3").arg(tr("Memoria global ")).arg((double)info.mem.totalGlobal / (1024 * 1024)).arg(tr("MB")) << endl;
	out << "\t" << QString("%1: %2 %3").arg(tr("Memoria compartida por bloque")).arg((double)info.mem.sharedPerBlock / 1024).arg(tr("KB")) << endl;
	out << "\t" << QString("%1: %2 %3").arg(tr("Pitch")).arg((double)info.mem.maxPitch / 1024).arg(tr("KB")) << endl;
	out << "\t" << QString("%1: %2 %3").arg(tr("Memoria constante")).arg((double)info.mem.totalConst / 1024).arg(tr("KB")) << endl;
	out << "\t" << QString("%1: %2").arg(tr("Texture Alignment")).arg(info.mem.textureAlignment) << endl;
	out << "\t" << QString("%1: %2").arg(tr("GPU Overlap")).arg(info.mem.gpuOverlap? tr("Yes"): tr("No")) << endl;
	out << endl;

	subtitle = tr("Información del rendimiento");
	out << subtitle << endl;
	for(int i = 0; i < subtitle.size(); i++)
		out << "-";
	out << endl;
	out << tr("Copiado de memoria") << endl;
	out << "\t" << tr("Host Pinned to Device") << ": ";
	if(info.band.copyHDPin == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.band.copyHDPin / 1024).arg(tr("MB/s")) << endl;
	out << "\t" << tr("Host Pageable to Device") << ": ";
	if(info.band.copyHDPage == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.band.copyHDPage / 1024).arg(tr("MB/s")) << endl;

	out << "\t" << tr("Device to Host Pinned") << ": ";
	if(info.band.copyDHPin == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.band.copyDHPin / 1024).arg(tr("MB/s")) << endl;
	out << "\t" << tr("Device to Host Pageable") << ": ";
	if(info.band.copyDHPage == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.band.copyDHPage / 1024).arg(tr("MB/s")) << endl;
	out << "\t" << tr("Device to Device") << ": ";
	if(info.band.copyDD == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.band.copyDD / 1024).arg(tr("MB/s")) << endl;
	out << tr("GPU Core Performance") << endl;
	out << "\t" << tr("Single-precision Float") << ": ";
	if(info.perf.calcFloat == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.perf.calcFloat / 1000).arg(tr("Mflop/s")) << endl;
	out << "\t" << tr("Double-precision Float") << ": ";
	if(((info.major > 1)) ||
		((info.major == 1) && (info.minor >= 3))) {
		if(info.perf.calcDouble == 0)
			out << "--" << endl;
		else
			out << QString("%1 %2").arg((double)info.perf.calcDouble / 1000).arg(tr("Mflop/s")) << endl;
	} else {
		out << tr("No soportado") << endl;
	}
	out << "\t" << tr("32-bit Integer") << ": ";
	if(info.perf.calcInteger32 == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.perf.calcInteger32 / 1000).arg(tr("Miop/s")) << endl;
	out << "\t" << tr("24-bit Integer") << ": ";
	if(info.perf.calcInteger24 == 0)
		out << "--" << endl;
	else
		out << QString("%1 %2").arg((double)info.perf.calcInteger24 / 1000).arg(tr("Miop/s")) << endl;
	out << endl;

	time_t t;
	time(&t);
	out << QString("%1: %2").arg(tr("Generated")).arg(ctime(&t)) << endl;
}

/*!
	\brief Export information to HTML file.
*/
void CZDialog::slotExportToHTML() {

	struct CZDeviceInfo info = deviceList[comboDevice->currentIndex()]->info();

	QString fileName = QFileDialog::getSaveFileName(this, tr("Save Text as..."),
		tr("%1.html").arg(tr(CZ_NAME_SHORT)), tr("HTML files (*.html *.htm);;All files (*.*)"));

	if(fileName.isEmpty())
		return;

	CZLog(CZLogLevelMid, "Export to HTML as %s", fileName);

	QFile file(fileName);
	if(!file.open(QFile::WriteOnly | QFile::Text)) {
		QMessageBox::warning(this, tr(CZ_NAME_SHORT),
			tr("Cannot write file %1:\n%2.").arg(fileName).arg(file.errorString()));
		return;
	}

	QTextStream out(&file);
	QString title = tr("%1 Report").arg(tr(CZ_NAME_SHORT));

	out << 	"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
		"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n"
		"<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"mul\" lang=\"mul\" dir=\"ltr\">\n"
		"<head>\n"
		"<title>" << title << "</title>\n"
		"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />\n"
		"<style type=\"text/css\">\n"

		"@charset \"utf-8\";\n"
		"body { font-size: 12px; font-family: Verdana, Arial, Helvetica, sans-serif; font-weight: normal; font-style: normal; }\n"
		"h1 { font-size: 15px; color: #690; }\n"
		"h2 { font-size: 13px; color: #690; }\n"
		"table { border-collapse: collapse; border: 1px solid #000; width: 500px; }\n"
		"th { background-color: #deb; text-align: left; }\n"
		"td { width: 50%; }\n"
		"a:link { color: #9c3; text-decoration: none; }\n"
		"a:visited { color: #690; text-decoration: none; }\n"
		"a:hover { color: #9c3; text-decoration: underline; }\n"
		"a:active { color: #9c3; text-decoration: underline; }\n"

		"</style>\n"
		"</head>\n"
		"<body style=\"background: #fff;\">\n";

	out << "<h1>" << title << "</h1>\n";
	out << "<p><small>";
	out << tr("<b>Version:</b> %1").arg(CZ_VERSION);
#ifdef CZ_VER_STATE
	out << tr(" <b>Built</b> %1 %2 ").arg(CZ_DATE).arg(CZ_TIME);
#endif//CZ_VER_STATE
	out << QString("<a href=\"%1\">%1</a><br/>\n").arg(CZ_ORG_URL_MAINPAGE);
	out << tr("<b>OS Version:</b> %1<br/>").arg(getOSVersion());
	out << "</small></p>\n";

	out << 	"<h2>" << tr("Core Information") << "</h2>\n"
		"<table border=\"1\">\n"
		"<tr><th>" << tr("Name") << "</th><td>" << info.deviceName << "</td></tr>\n"
		"<tr><th>" << tr("Compute Capability") << "</th><td>" << info.major << "." << info.minor << "</td></tr>\n"
		"<tr><th>" << tr("Clock Rate") << "</th><td>" << (double)info.core.clockRate / 1000 << " " << tr("MHz") << "</td></tr>\n";
	out << "<tr><th>" << tr("Multiprocessors") << "</th><td>";
	if(info.core.muliProcCount == 0)
		out << "<i>" << tr("Unknown") << "</i>";
	else
		out << info.core.muliProcCount;
	out << "</td></tr>\n";
	out <<	"<tr><th>" << tr("Warp Size") << "</th><td>" << info.core.SIMDWidth << "</td></tr>\n"
		"<tr><th>" << tr("Regs Per Block") << "</th><td>" << info.core.regsPerBlock << "</td></tr>\n"
		"<tr><th>" << tr("Threads Per Block") << "</th><td>" << info.core.maxThreadsPerBlock << "</td></tr>\n";
	out << "<tr><th>" << tr("Watchdog Enabled") << "</th><td>" << (info.core.watchdogEnabled? tr("Yes"): tr("No")) << "</td></tr>\n";
	out << "<tr><th>" << tr("Threads Dimentions") << "</th><td>" << info.core.maxThreadsDim[0] << " x " << info.core.maxThreadsDim[1] << " x " << info.core.maxThreadsDim[2] << "</td></tr>\n"
		"<tr><th>" << tr("Grid Dimentions") << "</th><td>" << info.core.maxGridSize[0] << " x " << info.core.maxGridSize[1] << " x " << info.core.maxGridSize[2] << "</td></tr>\n"
		"</table>\n";

	out << 	"<h2>" << tr("Memory Information") << "</h2>\n"
		"<table border=\"1\">\n"
		"<tr><th>" << tr("Total Global") << "</th><td>" << (double)info.mem.totalGlobal / (1024 * 1024) << " " << tr("MB") << "</td></tr>\n"
		"<tr><th>" << tr("Shared Per Block") << "</th><td>" << (double)info.mem.sharedPerBlock / 1024 << " " << tr("KB") << "</td></tr>\n"
		"<tr><th>" << tr("Pitch") << "</th><td>" << (double)info.mem.maxPitch / 1024 << " " << tr("KB") << "</td></tr>\n"
		"<tr><th>" << tr("Total Constant") << "</th><td>" << (double)info.mem.totalConst / 1024 << " " << tr("KB") << "</td></tr>\n"
		"<tr><th>" << tr("Texture Alignment") << "</th><td>" << info.mem.textureAlignment << "</td></tr>\n"
		"<tr><th>" << tr("GPU Overlap") << "</th><td>" << (info.mem.gpuOverlap? tr("Yes"): tr("No")) << "</td></tr>\n"
		"</table>\n";

	out << 	"<h2>" << tr("Performance Information") << "</h2>\n"
		"<table border=\"1\">\n"
		"<tr><th colspan=\"2\">" << tr("Memory Copy") << "</th></tr>\n"
		"<tr><th>" << tr("Host Pinned to Device") << "</th><td>";
		if(info.band.copyHDPin == 0)
			out << "--";
		else
			out << (double)info.band.copyHDPin / 1024 << " " << tr("MB/s");
		out << "</td></tr>\n"
		"<tr><th>" << tr("Host Pageable to Device") << "</th><td>";
		if(info.band.copyHDPage == 0)
			out << "--";
		else
			out << (double)info.band.copyHDPage / 1024 << " " << tr("MB/s");
		out << "</td></tr>\n"
		"<tr><th>" << tr("Device to Host Pinned") << "</th><td>";
		if(info.band.copyDHPin == 0)
			out << "--";
		else
			out << (double)info.band.copyDHPin / 1024 << " " << tr("MB/s");
		out << "</td></tr>\n"
		"<tr><th>" << tr("Device to Host Pageable") << "</th><td>";
		if(info.band.copyDHPage == 0)
			out << "--";
		else
			out << (double)info.band.copyDHPage / 1024 << " " << tr("MB/s");
		out << "</td></tr>\n"
		"<tr><th>" << tr("Device to Device") << "</th><td>";
		if(info.band.copyDD == 0)
			out << "--";
		else
			out << (double)info.band.copyDD / 1024 << " " << tr("MB/s");
		out << "</td></tr>\n"
		"<tr><th colspan=\"2\">" << tr("GPU Core Performance") << "</th></tr>\n"
		"<tr><th>" << tr("Single-precision Float") << "</th><td>";
		if(info.perf.calcFloat == 0)
			out << "--";
		else
			out << (double)info.perf.calcFloat / 1000 << " " << tr("Mflop/s");
		out << "</td></tr>\n"
		"<tr><th>" << tr("Double-precision Float") << "</th><td>";
		if(((info.major > 1)) ||
			((info.major == 1) && (info.minor >= 3))) {
			if(info.perf.calcDouble == 0)
				out << "--";
			else
				out << (double)info.perf.calcDouble / 1000 << " " << tr("Mflop/s");
		} else {
			out << "<i>" << tr("No soportado") << "</i>";
		}
		out << "</td></tr>\n"
		"<tr><th>" << tr("32-bit Integer") << "</th><td>";
		if(info.perf.calcInteger32 == 0)
			out << "--";
		else
			out << (double)info.perf.calcInteger32 / 1000 << " " << tr("Miop/s");
		out << "</td></tr>\n"
		"<tr><th>" << tr("24-bit Integer") << "</th><td>";
		if(info.perf.calcInteger24 == 0)
			out << "--";
		else
			out << (double)info.perf.calcInteger24 / 1000 << " " << tr("Miop/s");
		out << "</td></tr>\n"
		"</table>\n";

	time_t t;
	time(&t);
	out <<	"<p><small><b>" << tr("Generated") << ":</b> " << ctime(&t) << "</small></p>\n";

	out <<	"</body>\n"
		"</html>\n";
}

/*!
	\brief Start version reading procedure.
*/
void CZDialog::startGetHistoryHttp() {

	if(http == NULL) {
		//http = new QHttp(this);

		//connect(http, SIGNAL(done(bool)), this, SLOT(slotGetHistoryDone(bool)));
		//connect(http, SIGNAL(stateChanged(int)), this, SLOT(slotGetHistoryStateChanged(int)));

		//http->setHost(CZ_ORG_DOMAIN);
		//http->get("/history.txt");
	}

}

/*!
	\brief Clean up after version reading procedure.
*/
void CZDialog::cleanGetHistoryHttp() {

	if(http != NULL) {
		//disconnect(http, SIGNAL(done(bool)), this, SLOT(slotGetHistoryDone(bool)));
		//disconnect(http, SIGNAL(stateChanged(int)), this, SLOT(slotGetHistoryStateChanged(int)));

		delete http;
		http = NULL;
	}
}

/*!
	\brief HTTP operation result slot.
*/
void CZDialog::slotGetHistoryDone(
	bool error			/*!< HTTP operation error state. */
) {
	if(error) {
		CZLog(CZLogLevelWarning, "Get version request done with error %d: %s", http->error(), http->errorString());

		labelAppUpdate->setText(tr("Can't load version information.\n") + http->errorString());
	} else {
		CZLog(CZLogLevelMid, "Get version request done successfully");

		QString history(http->readAll().data());
		history.remove('\r');
		QStringList historyStrings(history.split("\n"));

		for(int i = 0; i < historyStrings.size(); i++) {
			CZLog(CZLogLevelLow, "%3d %s", i, historyStrings[i].toLocal8Bit().data());
		}

		QString lastVersion;
		QString downloadUrl;
		QString releaseNotes;

		bool validVersion = false;
		QString version;
		QString notes;
		QString url;

		QString nameVersion("version ");
		QString nameNotes("release-notes ");
		QString nameDownload = QString("download-") + CZ_OS_PLATFORM_STR + " ";

		for(int i = 0; i < historyStrings.size(); i++) {

			if(historyStrings[i].left(nameVersion.size()) == nameVersion) {

				if(validVersion) {
					downloadUrl = url;
					releaseNotes = notes;
					lastVersion = version;
				}

				version = historyStrings[i];
				version.remove(0, nameVersion.size());
				CZLog(CZLogLevelLow, "Version found: %s", version.toLocal8Bit().data());
				notes = "";
				url = "";
				validVersion = false;
			}
			if(historyStrings[i].left(nameNotes.size()) == nameNotes) {
				notes = historyStrings[i];
				notes.remove(0, nameNotes.size());
				CZLog(CZLogLevelLow, "Notes found: %s", notes.toLocal8Bit().data());
			}
			if(historyStrings[i].left(nameDownload.size()) == nameDownload) {
				url = historyStrings[i];
				url.remove(0, nameDownload.size());
				CZLog(CZLogLevelLow, "Valid URL found:", url.toLocal8Bit().data());
				validVersion = true;
			}
		}

		if(validVersion) {
			downloadUrl = url;
			releaseNotes = notes;
			lastVersion = version;
		}

		CZLog(CZLogLevelMid, "Last valid version: %s\n%s\n%s",
			lastVersion.toLocal8Bit().data(),
			releaseNotes.toLocal8Bit().data(),
			downloadUrl.toLocal8Bit().data());

		bool isNewest = true;
		bool isNonReleased = false;

		if(!lastVersion.isEmpty()) {

			QStringList versionNumbers = lastVersion.split('.');

			#define GEN_VERSION(major, minor) ((major * 10000) + minor)
			unsigned int myVersion = GEN_VERSION(CZ_VER_MAJOR, CZ_VER_MINOR);
			unsigned int lastVersion = GEN_VERSION(versionNumbers[0].toInt(), versionNumbers[1].toInt());;

			if(myVersion < lastVersion) {
				isNewest = false;
			} else if(myVersion == lastVersion) {
				isNewest = true;
#ifdef CZ_VER_BUILD
				if(CZ_VER_BUILD < versionNumbers[2].toInt()) {
					isNewest = false;
				}
#endif//CZ_VER_BUILD
			} else { // myVersion > lastVersion
				isNonReleased = true;
			}
		}

		if(isNewest) {
			if(isNonReleased) {
				labelAppUpdate->setText(tr("WARNING: You running prerelease version!"));
			} else {
				labelAppUpdate->setText(tr("No new version found."));
			}
		} else {
			QString updateString = QString("%1 <b>%2</b>.")
				.arg(tr("New version is available")).arg(lastVersion);
			if(!downloadUrl.isEmpty()) {
				updateString += QString("<br><a href=\"%1\">%2</a>")
					.arg(downloadUrl)
					.arg(tr("Download"));
			} else {
				updateString += QString("<br><a href=\"%1\">%2</a>")
					.arg(CZ_ORG_URL_MAINPAGE)
					.arg(tr("Main page"));
			}
			if(!releaseNotes.isEmpty()) {
				updateString += QString(" <a href=\"%1\">%2</a>")
					.arg(releaseNotes)
					.arg(tr("Release notes"));
			}
			labelAppUpdate->setText(updateString);
		}
	}
}

/*!
	\brief HTTP connection state change slot.
*/
void CZDialog::slotGetHistoryStateChanged(
	int state			/*!< Current state of HTTP link. */
) {
	CZLog(CZLogLevelLow, "Get version connection state changed to %d", state);

	if(state == QHttp::Unconnected) {
		CZLog(CZLogLevelLow, "Disconnected!");
	}
}

void CZDialog::setupUi()
    {
        this->resize(342, 341);
        QIcon icon;
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabInfo = new QTabWidget();
        tabInfo->setObjectName(QString::fromUtf8("tabInfo"));
        tabCore = new QWidget();
        tabCore->setObjectName(QString::fromUtf8("tabCore"));
        gridLayout_2 = new QGridLayout(tabCore);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        labelName = new QLabel(tabCore);
        labelName->setObjectName(QString::fromUtf8("labelName"));
        labelName->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelName, 0, 0, 1, 1);

        labelNameText = new QLabel(tabCore);
        labelNameText->setObjectName(QString::fromUtf8("labelNameText"));
        labelNameText->setFrameShape(QFrame::Panel);
        labelNameText->setFrameShadow(QFrame::Sunken);
        labelNameText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_2->addWidget(labelNameText, 0, 1, 1, 1);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);

        labelDeviceLogo = new QLabel(tabCore);
        labelDeviceLogo->setObjectName(QString::fromUtf8("labelDeviceLogo"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(labelDeviceLogo->sizePolicy().hasHeightForWidth());
        labelDeviceLogo->setSizePolicy(sizePolicy);
        labelDeviceLogo->setFrameShape(QFrame::Panel);
        labelDeviceLogo->setFrameShadow(QFrame::Sunken);
        labelDeviceLogo->setPixmap(QPixmap(QString::fromUtf8(":/img/logo-unknown.png")));
        labelDeviceLogo->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(labelDeviceLogo);


        verticalLayout_2->addLayout(horizontalLayout_2);

        verticalSpacer = new QSpacerItem(97, 88, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);


        gridLayout_2->addLayout(verticalLayout_2, 0, 2, 7, 1);

        labelCapability = new QLabel(tabCore);
        labelCapability->setObjectName(QString::fromUtf8("labelCapability"));
        labelCapability->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelCapability, 1, 0, 1, 1);

        labelCapabilityText = new QLabel(tabCore);
        labelCapabilityText->setObjectName(QString::fromUtf8("labelCapabilityText"));
        labelCapabilityText->setFrameShape(QFrame::Panel);
        labelCapabilityText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelCapabilityText, 1, 1, 1, 1);

        labelClock = new QLabel(tabCore);
        labelClock->setObjectName(QString::fromUtf8("labelClock"));
        labelClock->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelClock, 2, 0, 1, 1);

        labelClockText = new QLabel(tabCore);
        labelClockText->setObjectName(QString::fromUtf8("labelClockText"));
        labelClockText->setFrameShape(QFrame::Panel);
        labelClockText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelClockText, 2, 1, 1, 1);

        labelMultiProc = new QLabel(tabCore);
        labelMultiProc->setObjectName(QString::fromUtf8("labelMultiProc"));
        labelMultiProc->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelMultiProc, 3, 0, 1, 1);

        labelMultiProcText = new QLabel(tabCore);
        labelMultiProcText->setObjectName(QString::fromUtf8("labelMultiProcText"));
        labelMultiProcText->setFrameShape(QFrame::Panel);
        labelMultiProcText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelMultiProcText, 3, 1, 1, 1);

        labelWarp = new QLabel(tabCore);
        labelWarp->setObjectName(QString::fromUtf8("labelWarp"));
        labelWarp->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelWarp, 4, 0, 1, 1);

        labelWarpText = new QLabel(tabCore);
        labelWarpText->setObjectName(QString::fromUtf8("labelWarpText"));
        labelWarpText->setFrameShape(QFrame::Panel);
        labelWarpText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelWarpText, 4, 1, 1, 1);

        labelRegs = new QLabel(tabCore);
        labelRegs->setObjectName(QString::fromUtf8("labelRegs"));
        labelRegs->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelRegs, 5, 0, 1, 1);

        labelRegsText = new QLabel(tabCore);
        labelRegsText->setObjectName(QString::fromUtf8("labelRegsText"));
        labelRegsText->setFrameShape(QFrame::Panel);
        labelRegsText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelRegsText, 5, 1, 1, 1);

        labelThreads = new QLabel(tabCore);
        labelThreads->setObjectName(QString::fromUtf8("labelThreads"));
        labelThreads->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelThreads, 6, 0, 1, 1);

        labelThreadsText = new QLabel(tabCore);
        labelThreadsText->setObjectName(QString::fromUtf8("labelThreadsText"));
        labelThreadsText->setFrameShape(QFrame::Panel);
        labelThreadsText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelThreadsText, 6, 1, 1, 1);

        labelWatchdog = new QLabel(tabCore);
        labelWatchdog->setObjectName(QString::fromUtf8("labelWatchdog"));
        labelWatchdog->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelWatchdog, 7, 0, 1, 1);

        labelWatchdogText = new QLabel(tabCore);
        labelWatchdogText->setObjectName(QString::fromUtf8("labelWatchdogText"));
        labelWatchdogText->setFrameShape(QFrame::Panel);
        labelWatchdogText->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(labelWatchdogText, 7, 1, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        labelDimX = new QLabel(tabCore);
        labelDimX->setObjectName(QString::fromUtf8("labelDimX"));
        labelDimX->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(labelDimX, 0, 0, 1, 1);

        labelDimY = new QLabel(tabCore);
        labelDimY->setObjectName(QString::fromUtf8("labelDimY"));
        labelDimY->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(labelDimY, 0, 1, 1, 1);

        labelDimZ = new QLabel(tabCore);
        labelDimZ->setObjectName(QString::fromUtf8("labelDimZ"));
        labelDimZ->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(labelDimZ, 0, 2, 1, 1);

        labelThreadsDimTextX = new QLabel(tabCore);
        labelThreadsDimTextX->setObjectName(QString::fromUtf8("labelThreadsDimTextX"));
        labelThreadsDimTextX->setFrameShape(QFrame::Panel);
        labelThreadsDimTextX->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(labelThreadsDimTextX, 1, 0, 1, 1);

        labelThreadsDimTextY = new QLabel(tabCore);
        labelThreadsDimTextY->setObjectName(QString::fromUtf8("labelThreadsDimTextY"));
        labelThreadsDimTextY->setFrameShape(QFrame::Panel);
        labelThreadsDimTextY->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(labelThreadsDimTextY, 1, 1, 1, 1);

        labelThreadsDimTextZ = new QLabel(tabCore);
        labelThreadsDimTextZ->setObjectName(QString::fromUtf8("labelThreadsDimTextZ"));
        labelThreadsDimTextZ->setFrameShape(QFrame::Panel);
        labelThreadsDimTextZ->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(labelThreadsDimTextZ, 1, 2, 1, 1);

        labelGridDimTextX = new QLabel(tabCore);
        labelGridDimTextX->setObjectName(QString::fromUtf8("labelGridDimTextX"));
        labelGridDimTextX->setFrameShape(QFrame::Panel);
        labelGridDimTextX->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(labelGridDimTextX, 2, 0, 1, 1);

        labelGridDimTextY = new QLabel(tabCore);
        labelGridDimTextY->setObjectName(QString::fromUtf8("labelGridDimTextY"));
        labelGridDimTextY->setFrameShape(QFrame::Panel);
        labelGridDimTextY->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(labelGridDimTextY, 2, 1, 1, 1);

        labelGridDimTextZ = new QLabel(tabCore);
        labelGridDimTextZ->setObjectName(QString::fromUtf8("labelGridDimTextZ"));
        labelGridDimTextZ->setFrameShape(QFrame::Panel);
        labelGridDimTextZ->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(labelGridDimTextZ, 2, 2, 1, 1);


        gridLayout_2->addLayout(gridLayout, 8, 1, 3, 2);

        labelGridDim = new QLabel(tabCore);
        labelGridDim->setObjectName(QString::fromUtf8("labelGridDim"));
        labelGridDim->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelGridDim, 10, 0, 1, 1);

        verticalSpacer_2 = new QSpacerItem(20, 163, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer_2, 11, 1, 1, 1);

        labelThreadsDim = new QLabel(tabCore);
        labelThreadsDim->setObjectName(QString::fromUtf8("labelThreadsDim"));
        labelThreadsDim->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(labelThreadsDim, 9, 0, 1, 1);

        tabInfo->addTab(tabCore, QString());
        tabMemory = new QWidget();
        tabMemory->setObjectName(QString::fromUtf8("tabMemory"));
        gridLayout_3 = new QGridLayout(tabMemory);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        labelTotalGlobal = new QLabel(tabMemory);
        labelTotalGlobal->setObjectName(QString::fromUtf8("labelTotalGlobal"));
        labelTotalGlobal->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(labelTotalGlobal, 0, 0, 1, 1);

        labelTotalGlobalText = new QLabel(tabMemory);
        labelTotalGlobalText->setObjectName(QString::fromUtf8("labelTotalGlobalText"));
        labelTotalGlobalText->setFrameShape(QFrame::Panel);
        labelTotalGlobalText->setFrameShadow(QFrame::Sunken);
        labelTotalGlobalText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_3->addWidget(labelTotalGlobalText, 0, 1, 1, 1);

        labelShared = new QLabel(tabMemory);
        labelShared->setObjectName(QString::fromUtf8("labelShared"));
        labelShared->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(labelShared, 1, 0, 1, 1);

        labelSharedText = new QLabel(tabMemory);
        labelSharedText->setObjectName(QString::fromUtf8("labelSharedText"));
        labelSharedText->setFrameShape(QFrame::Panel);
        labelSharedText->setFrameShadow(QFrame::Sunken);
        labelSharedText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_3->addWidget(labelSharedText, 1, 1, 1, 1);

        labelPitch = new QLabel(tabMemory);
        labelPitch->setObjectName(QString::fromUtf8("labelPitch"));
        labelPitch->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(labelPitch, 2, 0, 1, 1);

        labelPitchText = new QLabel(tabMemory);
        labelPitchText->setObjectName(QString::fromUtf8("labelPitchText"));
        labelPitchText->setFrameShape(QFrame::Panel);
        labelPitchText->setFrameShadow(QFrame::Sunken);
        labelPitchText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_3->addWidget(labelPitchText, 2, 1, 1, 1);

        labelTotalConst = new QLabel(tabMemory);
        labelTotalConst->setObjectName(QString::fromUtf8("labelTotalConst"));
        labelTotalConst->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(labelTotalConst, 3, 0, 1, 1);

        labelTotalConstText = new QLabel(tabMemory);
        labelTotalConstText->setObjectName(QString::fromUtf8("labelTotalConstText"));
        labelTotalConstText->setFrameShape(QFrame::Panel);
        labelTotalConstText->setFrameShadow(QFrame::Sunken);
        labelTotalConstText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_3->addWidget(labelTotalConstText, 3, 1, 1, 1);

        labelTextureAlign = new QLabel(tabMemory);
        labelTextureAlign->setObjectName(QString::fromUtf8("labelTextureAlign"));
        labelTextureAlign->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(labelTextureAlign, 4, 0, 1, 1);

        labelTextureAlignmentText = new QLabel(tabMemory);
        labelTextureAlignmentText->setObjectName(QString::fromUtf8("labelTextureAlignmentText"));
        labelTextureAlignmentText->setFrameShape(QFrame::Panel);
        labelTextureAlignmentText->setFrameShadow(QFrame::Sunken);
        labelTextureAlignmentText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_3->addWidget(labelTextureAlignmentText, 4, 1, 1, 1);

        labelGpuOverlap = new QLabel(tabMemory);
        labelGpuOverlap->setObjectName(QString::fromUtf8("labelGpuOverlap"));
        labelGpuOverlap->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(labelGpuOverlap, 5, 0, 1, 1);

        labelGpuOverlapText = new QLabel(tabMemory);
        labelGpuOverlapText->setObjectName(QString::fromUtf8("labelGpuOverlapText"));
        labelGpuOverlapText->setFrameShape(QFrame::Panel);
        labelGpuOverlapText->setFrameShadow(QFrame::Sunken);
        labelGpuOverlapText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_3->addWidget(labelGpuOverlapText, 5, 1, 1, 1);

        verticalSpacer_3 = new QSpacerItem(132, 31, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer_3, 7, 0, 1, 2);

        tabInfo->addTab(tabMemory, QString());
        tabPerformance = new QWidget();
        tabPerformance->setObjectName(QString::fromUtf8("tabPerformance"));
        gridLayout_4 = new QGridLayout(tabPerformance);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        labelMemRate = new QLabel(tabPerformance);
        labelMemRate->setObjectName(QString::fromUtf8("labelMemRate"));
        labelMemRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelMemRate, 0, 0, 1, 1);

        labelRatePin = new QLabel(tabPerformance);
        labelRatePin->setObjectName(QString::fromUtf8("labelRatePin"));
        labelRatePin->setAlignment(Qt::AlignCenter);

        gridLayout_4->addWidget(labelRatePin, 0, 1, 1, 1);

        labelRatePage = new QLabel(tabPerformance);
        labelRatePage->setObjectName(QString::fromUtf8("labelRatePage"));
        labelRatePage->setAlignment(Qt::AlignCenter);

        gridLayout_4->addWidget(labelRatePage, 0, 2, 1, 1);

        labelHDRate = new QLabel(tabPerformance);
        labelHDRate->setObjectName(QString::fromUtf8("labelHDRate"));
        labelHDRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelHDRate, 1, 0, 1, 1);

        labelHDRatePinText = new QLabel(tabPerformance);
        labelHDRatePinText->setObjectName(QString::fromUtf8("labelHDRatePinText"));
        labelHDRatePinText->setFrameShape(QFrame::Panel);
        labelHDRatePinText->setFrameShadow(QFrame::Sunken);
        labelHDRatePinText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelHDRatePinText, 1, 1, 1, 1);

        labelHDRatePageText = new QLabel(tabPerformance);
        labelHDRatePageText->setObjectName(QString::fromUtf8("labelHDRatePageText"));
        labelHDRatePageText->setFrameShape(QFrame::Panel);
        labelHDRatePageText->setFrameShadow(QFrame::Sunken);
        labelHDRatePageText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelHDRatePageText, 1, 2, 1, 1);

        labelDHRate = new QLabel(tabPerformance);
        labelDHRate->setObjectName(QString::fromUtf8("labelDHRate"));
        labelDHRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelDHRate, 2, 0, 1, 1);

        labelDHRatePinText = new QLabel(tabPerformance);
        labelDHRatePinText->setObjectName(QString::fromUtf8("labelDHRatePinText"));
        labelDHRatePinText->setFrameShape(QFrame::Panel);
        labelDHRatePinText->setFrameShadow(QFrame::Sunken);
        labelDHRatePinText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelDHRatePinText, 2, 1, 1, 1);

        labelDHRatePageText = new QLabel(tabPerformance);
        labelDHRatePageText->setObjectName(QString::fromUtf8("labelDHRatePageText"));
        labelDHRatePageText->setFrameShape(QFrame::Panel);
        labelDHRatePageText->setFrameShadow(QFrame::Sunken);
        labelDHRatePageText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelDHRatePageText, 2, 2, 1, 1);

        labelDDRate = new QLabel(tabPerformance);
        labelDDRate->setObjectName(QString::fromUtf8("labelDDRate"));
        labelDDRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelDDRate, 3, 0, 1, 1);

        labelDDRateText = new QLabel(tabPerformance);
        labelDDRateText->setObjectName(QString::fromUtf8("labelDDRateText"));
        labelDDRateText->setFrameShape(QFrame::Panel);
        labelDDRateText->setFrameShadow(QFrame::Sunken);
        labelDDRateText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelDDRateText, 3, 1, 1, 2);

        labelCalcRate = new QLabel(tabPerformance);
        labelCalcRate->setObjectName(QString::fromUtf8("labelCalcRate"));
        labelCalcRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelCalcRate, 4, 0, 1, 1);

        labelFloatRate = new QLabel(tabPerformance);
        labelFloatRate->setObjectName(QString::fromUtf8("labelFloatRate"));
        labelFloatRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelFloatRate, 5, 0, 1, 1);

        labelFloatRateText = new QLabel(tabPerformance);
        labelFloatRateText->setObjectName(QString::fromUtf8("labelFloatRateText"));
        labelFloatRateText->setFrameShape(QFrame::Panel);
        labelFloatRateText->setFrameShadow(QFrame::Sunken);
        labelFloatRateText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelFloatRateText, 5, 1, 1, 2);

        labelDoubleRate = new QLabel(tabPerformance);
        labelDoubleRate->setObjectName(QString::fromUtf8("labelDoubleRate"));
        labelDoubleRate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelDoubleRate, 6, 0, 1, 1);

        labelDoubleRateText = new QLabel(tabPerformance);
        labelDoubleRateText->setObjectName(QString::fromUtf8("labelDoubleRateText"));
        labelDoubleRateText->setFrameShape(QFrame::Panel);
        labelDoubleRateText->setFrameShadow(QFrame::Sunken);
        labelDoubleRateText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelDoubleRateText, 6, 1, 1, 2);

        labelInt32Rate = new QLabel(tabPerformance);
        labelInt32Rate->setObjectName(QString::fromUtf8("labelInt32Rate"));
        labelInt32Rate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelInt32Rate, 7, 0, 1, 1);

        labelInt32RateText = new QLabel(tabPerformance);
        labelInt32RateText->setObjectName(QString::fromUtf8("labelInt32RateText"));
        labelInt32RateText->setFrameShape(QFrame::Panel);
        labelInt32RateText->setFrameShadow(QFrame::Sunken);
        labelInt32RateText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelInt32RateText, 7, 1, 1, 2);

        labelInt24Rate = new QLabel(tabPerformance);
        labelInt24Rate->setObjectName(QString::fromUtf8("labelInt24Rate"));
        labelInt24Rate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(labelInt24Rate, 8, 0, 1, 1);

        labelInt24RateText = new QLabel(tabPerformance);
        labelInt24RateText->setObjectName(QString::fromUtf8("labelInt24RateText"));
        labelInt24RateText->setFrameShape(QFrame::Panel);
        labelInt24RateText->setFrameShadow(QFrame::Sunken);
        labelInt24RateText->setTextInteractionFlags(Qt::LinksAccessibleByMouse);

        gridLayout_4->addWidget(labelInt24RateText, 8, 1, 1, 2);

        verticalSpacer_6 = new QSpacerItem(273, 110, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_4->addItem(verticalSpacer_6, 9, 0, 1, 3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        checkUpdateResults = new QCheckBox(tabPerformance);
        checkUpdateResults->setObjectName(QString::fromUtf8("checkUpdateResults"));
        checkUpdateResults->setChecked(true);

        horizontalLayout_4->addWidget(checkUpdateResults);

        horizontalSpacer_4 = new QSpacerItem(17, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_4);

        pushExport = new QPushButton(tabPerformance);
        pushExport->setObjectName(QString::fromUtf8("pushExport"));
        pushExport->setAutoDefault(false);

        horizontalLayout_4->addWidget(pushExport);


        gridLayout_4->addLayout(horizontalLayout_4, 10, 0, 1, 3);

        tabInfo->addTab(tabPerformance, QString());
        tabAbout = new QWidget();
        tabAbout->setObjectName(QString::fromUtf8("tabAbout"));
        verticalLayout_3 = new QVBoxLayout(tabAbout);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        labelAppLogo = new QLabel(tabAbout);
        labelAppLogo->setObjectName(QString::fromUtf8("labelAppLogo"));
        labelAppLogo->setPixmap(QPixmap(QString::fromUtf8(":/img/logo.png")));

        verticalLayout_3->addWidget(labelAppLogo);

        labelAppName = new QLabel(tabAbout);
        labelAppName->setObjectName(QString::fromUtf8("labelAppName"));

        verticalLayout_3->addWidget(labelAppName);

        labelAppVersion = new QLabel(tabAbout);
        labelAppVersion->setObjectName(QString::fromUtf8("labelAppVersion"));

        verticalLayout_3->addWidget(labelAppVersion);

        labelAppURL = new QLabel(tabAbout);
        labelAppURL->setObjectName(QString::fromUtf8("labelAppURL"));
        labelAppURL->setOpenExternalLinks(true);

        verticalLayout_3->addWidget(labelAppURL);

        labelAppAuthor = new QLabel(tabAbout);
        labelAppAuthor->setObjectName(QString::fromUtf8("labelAppAuthor"));

        verticalLayout_3->addWidget(labelAppAuthor);

        labelAppCopy = new QLabel(tabAbout);
        labelAppCopy->setObjectName(QString::fromUtf8("labelAppCopy"));
        labelAppCopy->setWordWrap(true);

        verticalLayout_3->addWidget(labelAppCopy);

        labelAppUpdate = new QLabel(tabAbout);
        labelAppUpdate->setObjectName(QString::fromUtf8("labelAppUpdate"));
        labelAppUpdate->setWordWrap(true);
        labelAppUpdate->setOpenExternalLinks(true);

        verticalLayout_3->addWidget(labelAppUpdate);

        verticalSpacer_4 = new QSpacerItem(20, 77, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer_4);

        //tabInfo->addTab(tabAbout, QString());

        verticalLayout->addWidget(tabInfo);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        comboDevice = new QComboBox();
        comboDevice->setObjectName(QString::fromUtf8("comboDevice"));
        comboDevice->setMinimumSize(QSize(200, 0));

        horizontalLayout->addWidget(comboDevice);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        pushOk = new QPushButton();
        pushOk->setObjectName(QString::fromUtf8("pushOk"));
        pushOk->setDefault(true);

        horizontalLayout->addWidget(pushOk);


        verticalLayout->addLayout(horizontalLayout);

        //QWidget::setTabOrder(comboDevice, pushOk);
        this->setTabOrder(comboDevice,pushOk);

        //QObject::connect(pushOk, SIGNAL(clicked()), this, SLOT(accept()));

        tabInfo->setCurrentIndex(0);

        QMetaObject::connectSlotsByName(this);
        setLayout(verticalLayout);
    } // setupUi

void CZDialog::retranslateUi()
    {
        labelName->setText(QApplication::translate("CZDialog", "Nombre", 0, QApplication::UnicodeUTF8));
        labelNameText->setText(QApplication::translate("CZDialog", "<name>", 0, QApplication::UnicodeUTF8));
        labelDeviceLogo->setText(QString());
        labelCapability->setText(QApplication::translate("CZDialog", "Capacidad computacional", 0, QApplication::UnicodeUTF8));
        labelCapabilityText->setText(QApplication::translate("CZDialog", "<capability>", 0, QApplication::UnicodeUTF8));
        labelClock->setText(QApplication::translate("CZDialog", "Velocidad de reloj", 0, QApplication::UnicodeUTF8));
        labelClockText->setText(QApplication::translate("CZDialog", "<clock MHz>", 0, QApplication::UnicodeUTF8));
        labelMultiProc->setText(QApplication::translate("CZDialog", "Multiprcesadores", 0, QApplication::UnicodeUTF8));
        labelMultiProcText->setText(QApplication::translate("CZDialog", "<procs>", 0, QApplication::UnicodeUTF8));
        labelWarp->setText(QApplication::translate("CZDialog", "Tamaño de Warp", 0, QApplication::UnicodeUTF8));
        labelWarpText->setText(QApplication::translate("CZDialog", "<size>", 0, QApplication::UnicodeUTF8));
        labelRegs->setText(QApplication::translate("CZDialog", "Registros por bloque", 0, QApplication::UnicodeUTF8));
        labelRegsText->setText(QApplication::translate("CZDialog", "<regs>", 0, QApplication::UnicodeUTF8));
        labelThreads->setText(QApplication::translate("CZDialog", "Hilos por bloque", 0, QApplication::UnicodeUTF8));
        labelThreadsText->setText(QApplication::translate("CZDialog", "<threads>", 0, QApplication::UnicodeUTF8));
        labelWatchdog->setText(QApplication::translate("CZDialog", "Watchdog Activado", 0, QApplication::UnicodeUTF8));
        labelWatchdogText->setText(QApplication::translate("CZDialog", "<yes/no>", 0, QApplication::UnicodeUTF8));
        labelDimX->setText(QApplication::translate("CZDialog", "X", 0, QApplication::UnicodeUTF8));
        labelDimY->setText(QApplication::translate("CZDialog", "Y", 0, QApplication::UnicodeUTF8));
        labelDimZ->setText(QApplication::translate("CZDialog", "Z", 0, QApplication::UnicodeUTF8));
        labelThreadsDimTextX->setText(QApplication::translate("CZDialog", "<X>", 0, QApplication::UnicodeUTF8));
        labelThreadsDimTextY->setText(QApplication::translate("CZDialog", "<Y>", 0, QApplication::UnicodeUTF8));
        labelThreadsDimTextZ->setText(QApplication::translate("CZDialog", "<Z>", 0, QApplication::UnicodeUTF8));
        labelGridDimTextX->setText(QApplication::translate("CZDialog", "<X>", 0, QApplication::UnicodeUTF8));
        labelGridDimTextY->setText(QApplication::translate("CZDialog", "<Y>", 0, QApplication::UnicodeUTF8));
        labelGridDimTextZ->setText(QApplication::translate("CZDialog", "<Z>", 0, QApplication::UnicodeUTF8));
        labelGridDim->setText(QApplication::translate("CZDialog", "Dimensiones de grid", 0, QApplication::UnicodeUTF8));
        labelThreadsDim->setText(QApplication::translate("CZDialog", "Dimensiones de hilo", 0, QApplication::UnicodeUTF8));
        tabInfo->setTabText(tabInfo->indexOf(tabCore), QApplication::translate("CZDialog", "Núcleo", 0, QApplication::UnicodeUTF8));
        labelTotalGlobal->setText(QApplication::translate("CZDialog", "Memoria global", 0, QApplication::UnicodeUTF8));
        labelTotalGlobalText->setText(QApplication::translate("CZDialog", "<size MB>", 0, QApplication::UnicodeUTF8));
        labelShared->setText(QApplication::translate("CZDialog", "Memoria compartida por bloque", 0, QApplication::UnicodeUTF8));
        labelSharedText->setText(QApplication::translate("CZDialog", "<size KB>", 0, QApplication::UnicodeUTF8));
        labelPitch->setText(QApplication::translate("CZDialog", "Pitch", 0, QApplication::UnicodeUTF8));
        labelPitchText->setText(QApplication::translate("CZDialog", "<size KB>", 0, QApplication::UnicodeUTF8));
        labelTotalConst->setText(QApplication::translate("CZDialog", "memoria constante", 0, QApplication::UnicodeUTF8));
        labelTotalConstText->setText(QApplication::translate("CZDialog", "<size KB>", 0, QApplication::UnicodeUTF8));
        labelTextureAlign->setText(QApplication::translate("CZDialog", "Memoria de textura", 0, QApplication::UnicodeUTF8));
        labelTextureAlignmentText->setText(QApplication::translate("CZDialog", "<size>", 0, QApplication::UnicodeUTF8));
        labelGpuOverlap->setText(QApplication::translate("CZDialog", "GPU Overlap", 0, QApplication::UnicodeUTF8));
        labelGpuOverlapText->setText(QApplication::translate("CZDialog", "<yes/no>", 0, QApplication::UnicodeUTF8));
        tabInfo->setTabText(tabInfo->indexOf(tabMemory), QApplication::translate("CZDialog", "Memoria", 0, QApplication::UnicodeUTF8));
        labelMemRate->setText(QApplication::translate("CZDialog", "Copiado de memoria", 0, QApplication::UnicodeUTF8));
        labelRatePin->setText(QApplication::translate("CZDialog", "Pinned", 0, QApplication::UnicodeUTF8));
        labelRatePage->setText(QApplication::translate("CZDialog", "Paginada", 0, QApplication::UnicodeUTF8));
        labelHDRate->setText(QApplication::translate("CZDialog", "De host a device", 0, QApplication::UnicodeUTF8));
        labelHDRatePinText->setText(QApplication::translate("CZDialog", "<rate MB/s>", 0, QApplication::UnicodeUTF8));
        labelHDRatePageText->setText(QApplication::translate("CZDialog", "<rate MB/s>", 0, QApplication::UnicodeUTF8));
        labelDHRate->setText(QApplication::translate("CZDialog", "De device a host", 0, QApplication::UnicodeUTF8));
        labelDHRatePinText->setText(QApplication::translate("CZDialog", "<rate MB/s>", 0, QApplication::UnicodeUTF8));
        labelDHRatePageText->setText(QApplication::translate("CZDialog", "<rate MB/s>", 0, QApplication::UnicodeUTF8));
        labelDDRate->setText(QApplication::translate("CZDialog", "Device a Device", 0, QApplication::UnicodeUTF8));
        labelDDRateText->setText(QApplication::translate("CZDialog", "<rate MB/s>", 0, QApplication::UnicodeUTF8));
        labelCalcRate->setText(QApplication::translate("CZDialog", "Rendimiento de núcleo GPU", 0, QApplication::UnicodeUTF8));
        labelFloatRate->setText(QApplication::translate("CZDialog", "Preciosión simple flotante", 0, QApplication::UnicodeUTF8));
        labelFloatRateText->setText(QApplication::translate("CZDialog", "<rate Mflop/s>", 0, QApplication::UnicodeUTF8));
        labelDoubleRate->setText(QApplication::translate("CZDialog", "Precisión doble flotante", 0, QApplication::UnicodeUTF8));
        labelDoubleRateText->setText(QApplication::translate("CZDialog", "<rate Mflop/s>", 0, QApplication::UnicodeUTF8));
        labelInt32Rate->setText(QApplication::translate("CZDialog", "Enteros 32-bits", 0, QApplication::UnicodeUTF8));
        labelInt32RateText->setText(QApplication::translate("CZDialog", "<rate Miop/s>", 0, QApplication::UnicodeUTF8));
        labelInt24Rate->setText(QApplication::translate("CZDialog", "Enteros 24-bits", 0, QApplication::UnicodeUTF8));
        labelInt24RateText->setText(QApplication::translate("CZDialog", "<rate Miop/s>", 0, QApplication::UnicodeUTF8));
        checkUpdateResults->setText(QApplication::translate("CZDialog", "Actualizar resultados en segundo plano", 0, QApplication::UnicodeUTF8));
        pushExport->setText(QApplication::translate("CZDialog", "&Export >>", 0, QApplication::UnicodeUTF8));
        tabInfo->setTabText(tabInfo->indexOf(tabPerformance), QApplication::translate("CZDialog", "Rendimiento", 0, QApplication::UnicodeUTF8));
        labelAppLogo->setText(QString());
        labelAppName->setText(QApplication::translate("CZDialog", "<name>", 0, QApplication::UnicodeUTF8));
        labelAppVersion->setText(QApplication::translate("CZDialog", "<version>", 0, QApplication::UnicodeUTF8));
        labelAppURL->setText(QApplication::translate("CZDialog", "<urls>", 0, QApplication::UnicodeUTF8));
        labelAppAuthor->setText(QApplication::translate("CZDialog", "<author>", 0, QApplication::UnicodeUTF8));
        labelAppCopy->setText(QApplication::translate("CZDialog", "<copyrights>", 0, QApplication::UnicodeUTF8));
        labelAppUpdate->setText(QApplication::translate("CZDialog", "<update>", 0, QApplication::UnicodeUTF8));
        tabInfo->setTabText(tabInfo->indexOf(tabAbout), QApplication::translate("CZDialog", "Creador", 0, QApplication::UnicodeUTF8));
        pushOk->setText(QApplication::translate("CZDialog", "&OK", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

void  CZDialog::createComponents(){}
void  CZDialog::instantiateElements(){}
void  CZDialog::layoutComponents(){}

void  CZDialog::loadOriginData(QStringList stringList){}
QList<QImage> CZDialog::getOriginData(){}
void CZDialog::loadResultData(QList<QImage> *imageList){}
QList<QImage> CZDialog::getResultData(){}

QStringList CZDialog::returnDevicesNames() {
	QStringList stringList;

	for(int i = 0; i < deviceList.size(); i++) {
		stringList.append(deviceList[i]->info().deviceName);
	}
	return stringList;
}

QString CZDialog::getHelpData(){
	return Constants::Instance()->getCzdiaglogHelpMessage();
}

QString CZDialog::getExecutionData(){

}

void CZDialog::saveImages(){

}

//Obtiene el String de información correspondiente para el panel
QString CZDialog::getAlgorythmHelpData(){

}

void CZDialog::showAlgorythmHelpMessage() {

}
