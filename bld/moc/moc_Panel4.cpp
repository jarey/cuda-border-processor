/****************************************************************************
** Meta object code from reading C++ file 'Panel4.h'
**
** Created: Thu Jan 31 23:29:05 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/ui/Panel4.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Panel4.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Panel4[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      17,    8,    7,    7, 0x0a,
      43,    7,    7,    7, 0x0a,
      60,    7,    7,    7, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_Panel4[] = {
    "Panel4\0\0mimeData\0getDrop(const QMimeData*)\0"
    "consultarCombo()\0showAlgorythmHelpMessage()\0"
};

const QMetaObject Panel4::staticMetaObject = {
    { &PanelBase::staticMetaObject, qt_meta_stringdata_Panel4,
      qt_meta_data_Panel4, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Panel4::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Panel4::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Panel4::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Panel4))
        return static_cast<void*>(const_cast< Panel4*>(this));
    return PanelBase::qt_metacast(_clname);
}

int Panel4::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = PanelBase::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: getDrop((*reinterpret_cast< const QMimeData*(*)>(_a[1]))); break;
        case 1: consultarCombo(); break;
        case 2: showAlgorythmHelpMessage(); break;
        default: ;
        }
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
