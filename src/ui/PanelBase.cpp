/*
 * PanelBase.cpp
 * Creado: 25/02/2012
 * Autor: jose
 * Descripción: panel con la funcionalidad base para los paneles concretos. Establece los mñetodos
 * abstractos que tendrán que ser obligatoriamente implementados por las subclases de cada panel concreto.
 */

#include "PanelBase.h"
#include <QtGui>
#include <QFileDialog>
#include <stdio.h>
#include <QImage>
#include <QGraphicsEllipseItem>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QPixmap>
#include <math.h>
#include <queue>
#include <QtDebug>
#include <qdialog.h>
#include "./src/common/Constants.h"
#include "./src/common/Controlador.h"
#include <QSizePolicy>

PanelBase::PanelBase() {
	this->matrixListOrigin = new QList<QImage>();
	this->matrixListResult = new QList<QImage>();
}

QString PanelBase::getApplicationMode() {
	return this->applicationMode;
}

void PanelBase::setApplicationMode(QString mode){
	this->applicationMode = mode;
}

//Setters
void PanelBase::setRunButtonNeeded(bool value){
	this->runButtonNeeded = value;
}
void PanelBase::setStopButtonNeeded(bool value){
	this->stopButtonNeeded = value;
}
void PanelBase::setOpenButtonNeeded(bool value){
	this->openButtonNeeded = value;
}
void PanelBase::setTimeButtonNeeded(bool value){
	this->timeButtonNeeded = value;
}
void PanelBase::setSaveButtonNeeded(bool value){
	this->saveButtonNeeded = value;
}
//Getters
bool PanelBase::getRunButtonNeeded(){
	return this->runButtonNeeded;
}
bool PanelBase::getStopButtonNeeded(){
	return this->stopButtonNeeded;
}
bool PanelBase::getOpenButtonNeeded(){
	return this->openButtonNeeded;
}
bool PanelBase::getTimeButtonNeeded(){
	return this->timeButtonNeeded;
}

bool PanelBase::getSaveButtonNeeded(){
	return this->saveButtonNeeded;
}

/**
 * Devuelve el listado de imágenes seleccionadas para los modos de selección múltiple
 * **/
QList<QImage>* PanelBase::getMatrixListOrigin() {
	return this->matrixListOrigin;
}

void PanelBase::setMatrixListOrigin(QList<QImage>* matrixListOrigin) {
	this->matrixListOrigin = matrixListOrigin;
}

/**
 * Devuelve el listado de imágenes procesadas
 * **/
QList<QImage>* PanelBase::getMatrixListResult() {
	return this->matrixListResult;
}

void PanelBase::setMatrixListResult(QList<QImage>* matrixListResult) {
	this->matrixListResult = matrixListResult;
}
