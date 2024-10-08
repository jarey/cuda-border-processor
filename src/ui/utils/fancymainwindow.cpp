//NOTA: Este código se ha utilizado y adaptado a partir del código fuente del software
//Qt Creator que incluye Qt y que se corresponde con licencia GNU, a continuación la licencia
//original.

/**************************************************************************
**
** This file is part of Qt Creator
**
** Copyright (c) 2011 Nokia Corporation and/or its subsidiary(-ies).
**
** Contact: Nokia Corporation (info@qt.nokia.com)
**
**
** GNU Lesser General Public License Usage
**
** This file may be used under the terms of the GNU Lesser General Public
** License version 2.1 as published by the Free Software Foundation and
** appearing in the file LICENSE.LGPL included in the packaging of this file.
** Please review the following information to ensure the GNU Lesser General
** Public License version 2.1 requirements will be met:
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights. These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** Other Usage
**
** Alternatively, this file may be used in accordance with the terms and
** conditions contained in a signed written agreement between you and Nokia.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
**
**************************************************************************/

#include "fancymainwindow.h"

#include <QtCore/QList>
#include <QtCore/QHash>

#include <QtGui/QAction>
#include <QtGui/QContextMenuEvent>
#include <QtGui/QMenu>
#include <QtGui/QDockWidget>
#include <QtCore/QSettings>

static const char lockedKeyC[] = "Locked";
static const char stateKeyC[] = "State";
static const int settingsVersion = 2;
static const char dockWidgetActiveState[] = "DockWidgetActiveState";

namespace Utils {

/*! \class Utils::FancyMainWindow

    \brief MainWindow with dock widgets and additional "lock" functionality
    (locking the dock widgets in place) and "reset layout" functionality.

    The dock actions and the additional actions should be accessible
    in a Window-menu.
*/

struct FancyMainWindowPrivate
{
    FancyMainWindowPrivate();

    bool m_locked;
    bool m_handleDockVisibilityChanges; //todo

    QAction m_menuSeparator1;
    QAction m_toggleLockedAction;
    QAction m_menuSeparator2;
    QAction m_resetLayoutAction;
    QDockWidget *m_toolBarDockWidget;
};

FancyMainWindowPrivate::FancyMainWindowPrivate() :
    m_locked(true),
    m_handleDockVisibilityChanges(true),
    m_menuSeparator1(0),
    m_toggleLockedAction(FancyMainWindow::tr("Locked"), 0),
    m_menuSeparator2(0),
    m_resetLayoutAction(FancyMainWindow::tr("Reset to Default Layout"), 0),
    m_toolBarDockWidget(0)
{
    m_toggleLockedAction.setCheckable(true);
    m_toggleLockedAction.setChecked(m_locked);
    m_menuSeparator1.setSeparator(true);
    m_menuSeparator2.setSeparator(true);
}

FancyMainWindow::FancyMainWindow(QWidget *parent) :
    QMainWindow(parent), d(new FancyMainWindowPrivate)
{
    connect(&d->m_toggleLockedAction, SIGNAL(toggled(bool)),
            this, SLOT(setLocked(bool)));
    connect(&d->m_resetLayoutAction, SIGNAL(triggered()),
            this, SIGNAL(resetLayout()));
}

FancyMainWindow::~FancyMainWindow()
{
    delete d;
}

QDockWidget *FancyMainWindow::addDockForWidget(QWidget *widget)
{
    QDockWidget *dockWidget = new QDockWidget(widget->windowTitle(), this);
    dockWidget->setWidget(widget);
    // Set an object name to be used in settings, derive from widget name
    const QString objectName = widget->objectName();
    if (objectName.isEmpty()) {
        dockWidget->setObjectName(QLatin1String("dockWidget") + QString::number(dockWidgets().size() + 1));
    } else {
        dockWidget->setObjectName(objectName + QLatin1String("DockWidget"));
    }
    connect(dockWidget->toggleViewAction(), SIGNAL(triggered()),
        this, SLOT(onDockActionTriggered()), Qt::QueuedConnection);
    connect(dockWidget, SIGNAL(visibilityChanged(bool)),
            this, SLOT(onDockVisibilityChange(bool)));
    connect(dockWidget, SIGNAL(topLevelChanged(bool)),
            this, SLOT(onTopLevelChanged()));
    dockWidget->setProperty(dockWidgetActiveState, true);
    updateDockWidget(dockWidget);
    return dockWidget;
}

void FancyMainWindow::updateDockWidget(QDockWidget *dockWidget)
{
    const QDockWidget::DockWidgetFeatures features =
            (d->m_locked) ? QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable
                       : QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable;
    if (dockWidget->property("managed_dockwidget").isNull()) { // for the debugger tool bar
        QWidget *titleBarWidget = dockWidget->titleBarWidget();
        if (d->m_locked && !titleBarWidget && !dockWidget->isFloating())
            titleBarWidget = new QWidget(dockWidget);
        else if ((!d->m_locked || dockWidget->isFloating()) && titleBarWidget) {
            delete titleBarWidget;
            titleBarWidget = 0;
        }
        dockWidget->setTitleBarWidget(titleBarWidget);
    }
    dockWidget->setFeatures(features);
}

void FancyMainWindow::onDockActionTriggered()
{
    QDockWidget *dw = qobject_cast<QDockWidget *>(sender()->parent());
    if (dw) {
        if (dw->isVisible())
            dw->raise();
    }
}

void FancyMainWindow::onDockVisibilityChange(bool visible)
{
    if (d->m_handleDockVisibilityChanges)
        sender()->setProperty(dockWidgetActiveState, visible);
}

void FancyMainWindow::onTopLevelChanged()
{
    updateDockWidget(qobject_cast<QDockWidget*>(sender()));
}

void FancyMainWindow::setTrackingEnabled(bool enabled)
{
    if (enabled) {
        d->m_handleDockVisibilityChanges = true;
        foreach (QDockWidget *dockWidget, dockWidgets())
            dockWidget->setProperty(dockWidgetActiveState, dockWidget->isVisible());
    } else {
        d->m_handleDockVisibilityChanges = false;
    }
}

void FancyMainWindow::setLocked(bool locked)
{
    d->m_locked = locked;
    foreach (QDockWidget *dockWidget, dockWidgets()) {
        updateDockWidget(dockWidget);
    }
}

void FancyMainWindow::hideEvent(QHideEvent *event)
{
    Q_UNUSED(event)
    handleVisibilityChanged(false);
}

void FancyMainWindow::showEvent(QShowEvent *event)
{
    Q_UNUSED(event)
    handleVisibilityChanged(true);
}

void FancyMainWindow::contextMenuEvent(QContextMenuEvent *event)
{
    QMenu *menu = createPopupMenu();
    menu->exec(event->globalPos());
    delete menu;
}

void FancyMainWindow::handleVisibilityChanged(bool visible)
{
    d->m_handleDockVisibilityChanges = false;
    foreach (QDockWidget *dockWidget, dockWidgets()) {
        if (dockWidget->isFloating()) {
            dockWidget->setVisible(visible
                && dockWidget->property(dockWidgetActiveState).toBool());
        }
    }
    if (visible)
        d->m_handleDockVisibilityChanges = true;
}

void FancyMainWindow::saveSettings(QSettings *settings) const
{
    QHash<QString, QVariant> hash = saveSettings();
    QHashIterator<QString, QVariant> it(hash);
    while (it.hasNext()) {
        it.next();
        settings->setValue(it.key(), it.value());
    }
}

void FancyMainWindow::restoreSettings(const QSettings *settings)
{
    QHash<QString, QVariant> hash;
    foreach (const QString &key, settings->childKeys()) {
        hash.insert(key, settings->value(key));
    }
    restoreSettings(hash);
}

QHash<QString, QVariant> FancyMainWindow::saveSettings() const
{
    QHash<QString, QVariant> settings;
    settings.insert(QLatin1String(stateKeyC), saveState(settingsVersion));
    settings.insert(QLatin1String(lockedKeyC), d->m_locked);
    foreach (QDockWidget *dockWidget, dockWidgets()) {
        settings.insert(dockWidget->objectName(),
                dockWidget->property(dockWidgetActiveState));
    }
    return settings;
}

void FancyMainWindow::restoreSettings(const QHash<QString, QVariant> &settings)
{
    QByteArray ba = settings.value(QLatin1String(stateKeyC), QByteArray()).toByteArray();
    if (!ba.isEmpty())
        restoreState(ba, settingsVersion);
    d->m_locked = settings.value(QLatin1String("Locked"), true).toBool();
    d->m_toggleLockedAction.setChecked(d->m_locked);
    foreach (QDockWidget *widget, dockWidgets()) {
        widget->setProperty(dockWidgetActiveState,
            settings.value(widget->objectName(), false));
    }
}

QList<QDockWidget *> FancyMainWindow::dockWidgets() const
{
    return qFindChildren<QDockWidget *>(this);
}

bool FancyMainWindow::isLocked() const
{
    return d->m_locked;
}

QMenu *FancyMainWindow::createPopupMenu()
{
    QMenu *menu = new QMenu(this);;
    QList<QDockWidget *> dockwidgets = qFindChildren<QDockWidget *>(this);
    for (int i = 0; i < dockwidgets.size(); ++i) {
        QDockWidget *dockWidget = dockwidgets.at(i);
        if (dockWidget->property("managed_dockwidget").isNull()
                && dockWidget->parentWidget() == this) {
            menu->addAction(dockwidgets.at(i)->toggleViewAction());
        }
    }
    menu->addAction(&d->m_menuSeparator1);
    menu->addAction(&d->m_toggleLockedAction);
    menu->addAction(&d->m_menuSeparator2);
    menu->addAction(&d->m_resetLayoutAction);
    return menu;
}

QAction *FancyMainWindow::menuSeparator1() const
{
    return &d->m_menuSeparator1;
}

QAction *FancyMainWindow::toggleLockedAction() const
{
    return &d->m_toggleLockedAction;
}

QAction *FancyMainWindow::menuSeparator2() const
{
    return &d->m_menuSeparator2;
}

QAction *FancyMainWindow::resetLayoutAction() const
{
    return &d->m_resetLayoutAction;
}

void FancyMainWindow::setDockActionsVisible(bool v)
{
    foreach (const QDockWidget *dockWidget, dockWidgets())
        dockWidget->toggleViewAction()->setVisible(v);
    d->m_toggleLockedAction.setVisible(v);
    d->m_menuSeparator1.setVisible(v);
    d->m_menuSeparator2.setVisible(v);
    d->m_resetLayoutAction.setVisible(v);
}

QDockWidget *FancyMainWindow::toolBarDockWidget() const
{
    return d->m_toolBarDockWidget;
}

void FancyMainWindow::setToolBarDockWidget(QDockWidget *dock)
{
    d->m_toolBarDockWidget = dock;
}

} // namespace Utils
