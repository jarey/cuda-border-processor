/****************************************************************************
** Meta object code from reading C++ file 'styledbar.h'
**
** Created: Thu Jan 31 23:29:08 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/ui/utils/styledbar.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'styledbar.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Utils__StyledBar[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_Utils__StyledBar[] = {
    "Utils::StyledBar\0"
};

const QMetaObject Utils::StyledBar::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_Utils__StyledBar,
      qt_meta_data_Utils__StyledBar, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Utils::StyledBar::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Utils::StyledBar::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Utils::StyledBar::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Utils__StyledBar))
        return static_cast<void*>(const_cast< StyledBar*>(this));
    return QWidget::qt_metacast(_clname);
}

int Utils::StyledBar::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
static const uint qt_meta_data_Utils__StyledSeparator[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_Utils__StyledSeparator[] = {
    "Utils::StyledSeparator\0"
};

const QMetaObject Utils::StyledSeparator::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_Utils__StyledSeparator,
      qt_meta_data_Utils__StyledSeparator, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Utils::StyledSeparator::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Utils::StyledSeparator::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Utils::StyledSeparator::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Utils__StyledSeparator))
        return static_cast<void*>(const_cast< StyledSeparator*>(this));
    return QWidget::qt_metacast(_clname);
}

int Utils::StyledSeparator::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE
