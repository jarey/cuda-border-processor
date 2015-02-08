/****************************************************************************
** Meta object code from reading C++ file 'botoneravertical.h'
**
** Created: Sun Jan 6 00:37:50 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/ui/botoneravertical.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'botoneravertical.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_BotoneraVertical[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      35,   18,   17,   17, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_BotoneraVertical[] = {
    "BotoneraVertical\0\0current,previous\0"
    "changePage(QListWidgetItem*,QListWidgetItem*)\0"
};

const QMetaObject BotoneraVertical::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_BotoneraVertical,
      qt_meta_data_BotoneraVertical, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &BotoneraVertical::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *BotoneraVertical::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *BotoneraVertical::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_BotoneraVertical))
        return static_cast<void*>(const_cast< BotoneraVertical*>(this));
    return QWidget::qt_metacast(_clname);
}

int BotoneraVertical::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: changePage((*reinterpret_cast< QListWidgetItem*(*)>(_a[1])),(*reinterpret_cast< QListWidgetItem*(*)>(_a[2]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
