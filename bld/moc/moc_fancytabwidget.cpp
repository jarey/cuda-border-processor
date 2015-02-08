/****************************************************************************
** Meta object code from reading C++ file 'fancytabwidget.h'
**
** Created: Thu Jan 31 23:29:09 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/ui/fancytabwidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'fancytabwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Core__Internal__FancyTab[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       1,   14, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // properties: name, type, flags
      31,   25, 0x87095103,

       0        // eod
};

static const char qt_meta_stringdata_Core__Internal__FancyTab[] = {
    "Core::Internal::FancyTab\0float\0fader\0"
};

const QMetaObject Core::Internal::FancyTab::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_Core__Internal__FancyTab,
      qt_meta_data_Core__Internal__FancyTab, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Core::Internal::FancyTab::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Core::Internal::FancyTab::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Core::Internal::FancyTab::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Core__Internal__FancyTab))
        return static_cast<void*>(const_cast< FancyTab*>(this));
    return QObject::qt_metacast(_clname);
}

int Core::Internal::FancyTab::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    
#ifndef QT_NO_PROPERTIES
     if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< float*>(_v) = fader(); break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: setFader(*reinterpret_cast< float*>(_v)); break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::ResetProperty) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 1;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}
static const uint qt_meta_data_Core__Internal__FancyTabBar[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      29,   28,   28,   28, 0x05,

 // slots: signature, parameters, type, tag, flags
      49,   28,   28,   28, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_Core__Internal__FancyTabBar[] = {
    "Core::Internal::FancyTabBar\0\0"
    "currentChanged(int)\0emitCurrentIndex()\0"
};

const QMetaObject Core::Internal::FancyTabBar::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_Core__Internal__FancyTabBar,
      qt_meta_data_Core__Internal__FancyTabBar, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Core::Internal::FancyTabBar::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Core::Internal::FancyTabBar::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Core::Internal::FancyTabBar::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Core__Internal__FancyTabBar))
        return static_cast<void*>(const_cast< FancyTabBar*>(this));
    return QWidget::qt_metacast(_clname);
}

int Core::Internal::FancyTabBar::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: currentChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: emitCurrentIndex(); break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void Core::Internal::FancyTabBar::currentChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
static const uint qt_meta_data_Core__Internal__FancyTabWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      38,   32,   31,   31, 0x05,
      62,   32,   31,   31, 0x05,

 // slots: signature, parameters, type, tag, flags
      82,   32,   31,   31, 0x0a,
     103,   32,   31,   31, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_Core__Internal__FancyTabWidget[] = {
    "Core::Internal::FancyTabWidget\0\0index\0"
    "currentAboutToShow(int)\0currentChanged(int)\0"
    "setCurrentIndex(int)\0showWidget(int)\0"
};

const QMetaObject Core::Internal::FancyTabWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_Core__Internal__FancyTabWidget,
      qt_meta_data_Core__Internal__FancyTabWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Core::Internal::FancyTabWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Core::Internal::FancyTabWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Core::Internal::FancyTabWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Core__Internal__FancyTabWidget))
        return static_cast<void*>(const_cast< FancyTabWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int Core::Internal::FancyTabWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: currentAboutToShow((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: currentChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: setCurrentIndex((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: showWidget((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void Core::Internal::FancyTabWidget::currentAboutToShow(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Core::Internal::FancyTabWidget::currentChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
