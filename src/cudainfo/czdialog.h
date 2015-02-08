/*
 * czdialog.cpp
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: panel de mostrado de la información de la GPU. Esta clase reutiliza código
 *  obtenido del proyecto open-source CUDA Z, adaptando dicho código para integrarlo en el proyecto,
 *  haciendo uso de la licencia GPL.
 */

#ifndef CZ_DIALOG_H
#define CZ_DIALOG_H

#include <QSplashScreen>
#include <QTimer>
#include <QHttp>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QTabWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

#include "czdeviceinfo.h"
#include "cudainfo.h"
#include "./src/ui/PanelBase.h"

extern void wait(int n); // implemented in main.cpp

class CZSplashScreen: public QSplashScreen {
	Q_OBJECT

public:
	explicit CZSplashScreen(const QPixmap &pixmap = QPixmap(), int maxLines = 1, Qt::WindowFlags f = 0);
	CZSplashScreen(QWidget *parent, const QPixmap &pixmap = QPixmap(), int maxLines = 1, Qt::WindowFlags f = 0);
	virtual ~CZSplashScreen();

	void setMaxLines(int maxLines);
	int maxLines();

public slots:
	void showMessage(const QString &message, int alignment = Qt::AlignLeft, const QColor &color = Qt::black);
	void clearMessage();

private:
	QString m_message;
	int m_maxLines;
	int m_lines;
	int m_alignment;
	QColor m_color;

	void deleteTop(int lines = 1);
};

extern CZSplashScreen *splash;

class CZDialog: public PanelBase {
	Q_OBJECT

public:
	CZDialog();
	~CZDialog();
	void setupUi();
	void retranslateUi();
	void  loadOriginData(QStringList stringList);
	QList<QImage>  getOriginData();
	void  loadResultData(QList<QImage> *imageList);
	QList<QImage>  getResultData();
	QStringList returnDevicesNames();
	void saveImages();
	QString getAlgorythmHelpData();


private:
	QList<CZCudaDeviceInfo*> deviceList;
	QTimer *updateTimer;
	QHttp *http;
    QVBoxLayout *verticalLayout;
    QTabWidget *tabInfo;
    QWidget *tabCore;
    QGridLayout *gridLayout_2;
    QLabel *labelName;
    QLabel *labelNameText;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer_2;
    QLabel *labelDeviceLogo;
    QSpacerItem *verticalSpacer;
    QLabel *labelCapability;
    QLabel *labelCapabilityText;
    QLabel *labelClock;
    QLabel *labelClockText;
    QLabel *labelMultiProc;
    QLabel *labelMultiProcText;
    QLabel *labelWarp;
    QLabel *labelWarpText;
    QLabel *labelRegs;
    QLabel *labelRegsText;
    QLabel *labelThreads;
    QLabel *labelThreadsText;
    QLabel *labelWatchdog;
    QLabel *labelWatchdogText;
    QGridLayout *gridLayout;
    QLabel *labelDimX;
    QLabel *labelDimY;
    QLabel *labelDimZ;
    QLabel *labelThreadsDimTextX;
    QLabel *labelThreadsDimTextY;
    QLabel *labelThreadsDimTextZ;
    QLabel *labelGridDimTextX;
    QLabel *labelGridDimTextY;
    QLabel *labelGridDimTextZ;
    QLabel *labelGridDim;
    QSpacerItem *verticalSpacer_2;
    QLabel *labelThreadsDim;
    QWidget *tabMemory;
    QGridLayout *gridLayout_3;
    QLabel *labelTotalGlobal;
    QLabel *labelTotalGlobalText;
    QLabel *labelShared;
    QLabel *labelSharedText;
    QLabel *labelPitch;
    QLabel *labelPitchText;
    QLabel *labelTotalConst;
    QLabel *labelTotalConstText;
    QLabel *labelTextureAlign;
    QLabel *labelTextureAlignmentText;
    QLabel *labelGpuOverlap;
    QLabel *labelGpuOverlapText;
    QSpacerItem *verticalSpacer_3;
    QWidget *tabPerformance;
    QGridLayout *gridLayout_4;
    QLabel *labelMemRate;
    QLabel *labelRatePin;
    QLabel *labelRatePage;
    QLabel *labelHDRate;
    QLabel *labelHDRatePinText;
    QLabel *labelHDRatePageText;
    QLabel *labelDHRate;
    QLabel *labelDHRatePinText;
    QLabel *labelDHRatePageText;
    QLabel *labelDDRate;
    QLabel *labelDDRateText;
    QLabel *labelCalcRate;
    QLabel *labelFloatRate;
    QLabel *labelFloatRateText;
    QLabel *labelDoubleRate;
    QLabel *labelDoubleRateText;
    QLabel *labelInt32Rate;
    QLabel *labelInt32RateText;
    QLabel *labelInt24Rate;
    QLabel *labelInt24RateText;
    QSpacerItem *verticalSpacer_6;
    QHBoxLayout *horizontalLayout_4;
    QCheckBox *checkUpdateResults;
    QSpacerItem *horizontalSpacer_4;
    QPushButton *pushExport;
    QWidget *tabAbout;
    QVBoxLayout *verticalLayout_3;
    QLabel *labelAppLogo;
    QLabel *labelAppName;
    QLabel *labelAppVersion;
    QLabel *labelAppURL;
    QLabel *labelAppAuthor;
    QLabel *labelAppCopy;
    QLabel *labelAppUpdate;
    QSpacerItem *verticalSpacer_4;
    QHBoxLayout *horizontalLayout;
    QComboBox *comboDevice;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushOk;
	QPushButton *botonPrueba;

	void readCudaDevices();
	void freeCudaDevices();
	int getCudaDeviceNumber();

	void setupDeviceList();
	void setupDeviceInfo(int dev);
	void setupCoreTab(struct CZDeviceInfo &info);
	void setupMemoryTab(struct CZDeviceInfo &info);
	void setupPerformanceTab(struct CZDeviceInfo &info);
	void setupAboutTab();

	QString getOSVersion();

	void startGetHistoryHttp();
	void cleanGetHistoryHttp();
	void createComponents();
	void instantiateElements();
	void layoutComponents();
	QString getHelpData();
	QString getExecutionData();

private slots:
	void slotShowDevice(int index);
	void slotUpdatePerformance(int index);
	void slotUpdateTimer();
	void slotExportToText();
	void slotExportToHTML();
	void slotGetHistoryDone(bool error);
	void slotGetHistoryStateChanged(int state);
	void showAlgorythmHelpMessage();
};

#endif//CZ_DIALOG_H
