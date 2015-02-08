/****************************************************************************
** Meta object code from reading C++ file 'manhattanstyle.h'
**
** Created: Thu Jan 31 23:29:07 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/ui/utils/manhattanstyle.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'manhattanstyle.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ManhattanStyle[] = {

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
      49,   22,   16,   15, 0x09,

       0        // eod
};

static const char qt_meta_stringdata_ManhattanStyle[] = {
    "ManhattanStyle\0\0QIcon\0standardIcon,option,widget\0"
    "standardIconImplementation(StandardPixmap,const QStyleOption*,const QW"
    "idget*)\0"
};

const QMetaObject ManhattanStyle::staticMetaObject = {
    { &QProxyStyle::staticMetaObject, qt_meta_stringdata_ManhattanStyle,
      qt_meta_data_ManhattanStyle, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ManhattanStyle::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ManhattanStyle::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ManhattanStyle::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ManhattanStyle))
        return static_cast<void*>(const_cast< ManhattanStyle*>(this));
    return QProxyStyle::qt_metacast(_clname);
}

int ManhattanStyle::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QProxyStyle::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: { QIcon _r = standardIconImplementation((*reinterpret_cast< StandardPixmap(*)>(_a[1])),(*reinterpret_cast< const QStyleOption*(*)>(_a[2])),(*reinterpret_cast< const QWidget*(*)>(_a[3])));
            if (_a[0]) *reinterpret_cast< QIcon*>(_a[0]) = _r; }  break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
