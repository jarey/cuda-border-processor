/*
 * CImage.h
 *
 *  Creado: 20/04/2012
 *  Autor: jose
 *  Descripción: Clase de implementación del proceso de Canny, aporta funcionalidad adicional encapsulada a partir del tipo de dato QImage de Qt.
 */


#ifndef CIMAGE_H
#define CIMAGE_H

#include "globals.h"

class CImage
{
public:
	//Atributos de clase
	//Ancho y alto de la imagen.
    uint mWidth, mHeight;
    //Imagen original e imagen de trabajo.
    QImage * mOriginalImage, * mImage;
    //Matriz de supressión de no-maximos.
    CMatD * mSuppressed;

    //Constructores de la clase
    CImage(uint w, uint h);
    CImage(QString file);
    CImage(QImage image);
    //Destructor de la clase.
    ~CImage();

    //Métodos de la clase, procesamiento de canny,supressión de no-maximos y ejecución de histéresis.
    void canny(double blurSigma, bool useR, bool useG, bool useB);
    void useSuppressed();
    void useHysteresis(double thresholdLow, double thresholdHigh);

    //Métodos de ejecución de proceso de supresión de no-maximos, ejecución de histéresis y creación de filtro gaussiano.
    CMatD * suppression(CMatD& grad, CMatrix<int>& thetaClamped);
    CMatrix<int> * hysteresis(CMatD& grad, double thresholdLow, double thresholdHigh);
    CMatD gaussianFilter(double sigma);
};

#endif // CIMAGE_H
