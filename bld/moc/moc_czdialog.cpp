/****************************************************************************
** Meta object code from reading C++ file 'czdialog.h'
**
** Created: Thu Jan 31 23:29:10 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/cudainfo/czdialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'czdialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_CZSplashScreen[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      40,   16,   15,   15, 0x0a,
      90,   72,   15,   15, 0x2a,
     123,  115,   15,   15, 0x2a,
     144,   15,   15,   15, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_CZSplashScreen[] = {
    "CZSplashScreen\0\0message,alignment,color\0"
    "showMessage(QString,int,QColor)\0"
    "message,alignment\0showMessage(QString,int)\0"
    "message\0showMessage(QString)\0"
    "clearMessage()\0"
};

const QMetaObject CZSplashScreen::staticMetaObject = {
    { &QSplashScreen::staticMetaObject, qt_meta_stringdata_CZSplashScreen,
      qt_meta_data_CZSplashScreen, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CZSplashScreen::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CZSplashScreen::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CZSplashScreen::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CZSplashScreen))
        return static_cast<void*>(const_cast< CZSplashScreen*>(this));
    return QSplashScreen::qt_metacast(_clname);
}

int CZSplashScreen::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QSplashScreen::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: showMessage((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< const QColor(*)>(_a[3]))); break;
        case 1: showMessage((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 2: showMessage((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 3: clearMessage(); break;
        default: ;
        }
        _id -= 4;
    }
    return _id;
}
static const uint qt_meta_data_CZDialog[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      16,   10,    9,    9, 0x08,
      36,   10,    9,    9, 0x08,
      63,    9,    9,    9, 0x08,
      81,    9,    9,    9, 0x08,
     100,    9,    9,    9, 0x08,
     125,  119,    9,    9, 0x08,
     156,  150,    9,    9, 0x08,
     188,    9,    9,    9, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CZDialog[] = {
    "CZDialog\0\0index\0slotShowDevice(int)\0"
    "slotUpdatePerformance(int)\0slotUpdateTimer()\0"
    "slotExportToText()\0slotExportToHTML()\0"
    "error\0slotGetHistoryDone(bool)\0state\0"
    "slotGetHistoryStateChanged(int)\0"
    "showAlgorythmHelpMessage()\0"
};

const QMetaObject CZDialog::staticMetaObject = {
    { &PanelBase::staticMetaObject, qt_meta_stringdata_CZDialog,
      qt_meta_data_CZDialog, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CZDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CZDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CZDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CZDialog))
        return static_cast<void*>(const_cast< CZDialog*>(this));
    return PanelBase::qt_metacast(_clname);
}

int CZDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = PanelBase::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: slotShowDevice((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: slotUpdatePerformance((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: slotUpdateTimer(); break;
        case 3: slotExportToText(); break;
        case 4: slotExportToHTML(); break;
        case 5: slotGetHistoryDone((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: slotGetHistoryStateChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: showAlgorythmHelpMessage(); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
