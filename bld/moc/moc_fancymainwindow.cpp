/****************************************************************************
** Meta object code from reading C++ file 'fancymainwindow.h'
**
** Created: Thu Jan 31 23:29:06 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/ui/utils/fancymainwindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'fancymainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Utils__FancyMainWindow[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      24,   23,   23,   23, 0x05,

 // slots: signature, parameters, type, tag, flags
      45,   38,   23,   23, 0x0a,
      63,   61,   23,   23, 0x0a,
      91,   23,   23,   23, 0x08,
     115,   23,   23,   23, 0x08,
     144,   23,   23,   23, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_Utils__FancyMainWindow[] = {
    "Utils::FancyMainWindow\0\0resetLayout()\0"
    "locked\0setLocked(bool)\0v\0"
    "setDockActionsVisible(bool)\0"
    "onDockActionTriggered()\0"
    "onDockVisibilityChange(bool)\0"
    "onTopLevelChanged()\0"
};

const QMetaObject Utils::FancyMainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_Utils__FancyMainWindow,
      qt_meta_data_Utils__FancyMainWindow, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Utils::FancyMainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Utils::FancyMainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Utils::FancyMainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Utils__FancyMainWindow))
        return static_cast<void*>(const_cast< FancyMainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int Utils::FancyMainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: resetLayout(); break;
        case 1: setLocked((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: setDockActionsVisible((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: onDockActionTriggered(); break;
        case 4: onDockVisibilityChange((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: onTopLevelChanged(); break;
        default: ;
        }
        _id -= 6;
    }
    return _id;
}

// SIGNAL 0
void Utils::FancyMainWindow::resetLayout()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
