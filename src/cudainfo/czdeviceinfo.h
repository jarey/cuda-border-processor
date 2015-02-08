/*!
	\file czdeviceinfo.h
	\brief CUDA device information header file.
	\author AG
*/

#ifndef CZ_DEVICEINFO_H
#define CZ_DEVICEINFO_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

#include "cudainfo.h"

class CZCudaDeviceInfo;

class CZUpdateThread: public QThread {
	Q_OBJECT

public:
	CZUpdateThread(CZCudaDeviceInfo *info, QObject *parent = 0);
	~CZUpdateThread();

	void testPerformance(int index);
	void waitPerformance();

signals:
	void testedPerformance(int index);

protected:
	void run();

private:
	QMutex mutex;
	QWaitCondition newLoop;
	QWaitCondition readyForWork;
	QWaitCondition testStart;
	QWaitCondition testFinish;
	CZCudaDeviceInfo *info;

	int index;
	bool abort;
	bool deviceReady;
	bool testRunning;
};

class CZCudaDeviceInfo: public QObject {
	Q_OBJECT

public:
	CZCudaDeviceInfo(int devNum, QObject *parent = 0);
	~CZCudaDeviceInfo();

	int readInfo();
	int prepareDevice();
	int updateInfo();
	int cleanDevice();

	struct CZDeviceInfo &info();
//	CZUpdateThread *thread();

	void testPerformance(int index);
	void waitPerformance();

signals:
	void testedPerformance(int index);

private:
	struct CZDeviceInfo _info;
	CZUpdateThread *_thread;
};

#endif//CZ_DEVICEINFO_H
