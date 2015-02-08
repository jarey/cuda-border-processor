/*!
	\file log.cpp
	\brief Logging source file.
	\author AG
*/

//#include <QDebug>
#include <QByteArray>

#include <stdarg.h>

#include "log.h"

#define CZ_LOG_BUFFER_LENGTH		4096

/*
	\brief Logging function.
*/
void CZLog(
	CZLogLevel level,		/*!< Log level value. */
	char *fmt,					/*!< printf()-like format string. */
	...							/*!< Additional arguments for printout. */
) {
    char buf[CZ_LOG_BUFFER_LENGTH];
    buf[CZ_LOG_BUFFER_LENGTH - 1] = '\0';

#ifdef QT_NO_DEBUG
	if(level > CZLogLevelHigh) {
		return;
	}
#endif

	va_list ap;
    va_start(ap, fmt);
    if(fmt)
        qvsnprintf(buf, CZ_LOG_BUFFER_LENGTH - 1, fmt, ap);
    va_end(ap);

	QtMsgType type;
	switch(level) {
	case CZLogLevelFatal:
		type = QtFatalMsg;
		break;
	case CZLogLevelError:
		type = QtCriticalMsg;
		break;
	case CZLogLevelWarning:
		type = QtWarningMsg;
		break;
	default:
		type = QtDebugMsg;
		break;
	}

	qt_message_output(type, buf);
}
