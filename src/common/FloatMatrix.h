/*
 * FloatMatrix.h : Fichero de definición de la clase que encapsula una matriz de float con sus
 * 					operaciones para ser utilizada a modo de contenedor.
 *
 *  Created on: 22/03/2012
 *      Author: jose
 */

#ifndef FloatMatrix_H
#define	FloatMatrix_H

#include <stdio.h>
#include <math.h>
#include <QImage>

using namespace std;

class FloatMatrix{

public:
	FloatMatrix();
    //Constructor con argumentos de ancho y alto
    FloatMatrix(float widht, float height);
    //Constructor a partir de una QImage
    FloatMatrix(QImage image);
    //Destructor
	virtual ~FloatMatrix();
	//Operaciones para matrix:
	void insertPos(int x,int y, float floatValue);
	float getPos(int x, int y);

    //Setters y getters para los atributos.
        //getters:
    int getWidth();
    int getHeight();
    int getElementsNumber();
    float *getElements();
        //setters:
    void setWidth(float width);
    void setHeight(float height);
    void setElements(float *elements);

private:
    //Atributos
    int widht;
    int height;
    float *elements;
};

#endif	/* FloatMatrix_H */
