/*
 * globals.h
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: clase auxiliar para agrupamiento de inclusiones de clases y librerias para el proceso de Canny.
 */


#ifndef GLOBALS_H
#define GLOBALS_H

typedef unsigned int uint;

#include <QDebug>

#include <QImage>
#include <QString>
#include <QMessageBox>
#include <QTime>

#include <cmath>
#include <queue>

//Librerias estandard para pila y par.
using std::queue;
using std::pair;

//Definición de plantillas para matrices genéricas.
#include "CMatrix.h"

#endif // GLOBALS_H
