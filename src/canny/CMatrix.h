/*
 * CMatrix.h
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: clase de definición e implementación para la instanciación y trabajo mediante filtrado de matrices genéricas.
 *  			 Se incluyen métodos para procesado mediante filtros sobre la matriz definida y cálculo del gradiente.
 */

#ifndef CMATRIX_H
#define CMATRIX_H
//inclusión de elementos generales para canny.
#include "globals.h"

template<typename T> class CArray
{
public:
	//Tamaño de la matriz y elementos genérico.
    uint mSize;
    T * mItems;

    //Constructor y destructor
    CArray(uint size);
    ~CArray();
    //Definción de poerador genérico.
    T& operator[](uint index) { return mItems[index]; }
    //Operador división genérico.
    void operator/=(T quotient);
    //operador genérico suma.
    T sum();
};

/*
 * Clase: CMatrix
 * Propósito: Clase que encapsula las funcionalidades de convolución de filtros y propiedades genéricas de matriz
 * 			sobre tipos de datos genéricos al ser definida con plantillas.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> class CMatrix
{
public:
	//Variables de clase: ancho y alto, matriz genérica.
    uint mHeight, mWidth;
    CArray<T> ** mRows;

    //Constructores de clase
    CMatrix(CMatrix<T> * copyFrom);
    CMatrix(uint height, uint width);
    CMatrix(uint height, uint width, T initialValue);
    CMatrix(QImage * image, bool useR = true, bool useG = true, bool useB = true);
    //Destructor de clase.
    ~CMatrix();

    //Operadores sobre plantillas para matriz genérica.
    CArray<T> & operator[](uint index) { return *mRows[index]; } 
    T& at(uint x, uint y) { return (*mRows[x]).operator [](y); }     // Fila, columna.
    void set(uint x, uint y, T value) { return mRows[x][y]; }        // Fila, columna.

    //Operador de división y de adicción genérico.
    void operator/=(T quotient);
    void operator+=(CMatrix<T>& summand);
    //Cálculo de cuadrado y de raíz cuadrada.
    void squareElementsInPlace();
    void squareRootElementsInPlace();
    //Cálculo de arcotangente.
    static CMatrix<T> * atan2(CMatrix<T>& y, CMatrix<T>& x);
    //Cálculo de suma.
    T sum();
    //Filtrado de la matriz mediante convolusiócn de filtro de entrada.
    CMatrix<T> * filterBy(CMatrix<T>& kernel);
    //Copiado a nueva imagen.
    QImage * toNewImage(bool rescale = true);
    void debugPrint();

};

// Declaración de tipos de datos.
typedef CMatrix<double> CMatD;

/*
 * Constructor: CArray
 * @argument size: tamño de unsigned int.
 * Propósito: Constructor de matriz genérica para tipo de tado unsigned int.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CArray<T>::CArray(uint size)
        : mSize(size)
{
    mItems = new T[mSize];
}

/*
 * Destructor: CArray
 * Propósito: elimina la matrix de memoria.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CArray<T>::~CArray()
{
    delete [] mItems;
}

/*
 * Método: operator/=
 * @argument size: tamño de unsigned int.
 * Propósito: Realiza la división de los elementos de la matriz por un valor de un tipo de dato especificado.
 * 			  (operador división)
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> void CArray<T>::operator/=(T quotient)
{
    for(uint i = 0; i < mSize; i++)
        mItems[i] /= quotient;
}

/*
 * Método: sum
 * @argument size: tamño de unsigned int.
 * Propósito: Realiza la suma de los elementos de la matriz
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> T CArray<T>::sum()
{
    T r = 0;     // El tipo de datos debe tener un 0 declarado.
    for(uint i = 0; i < mSize; i++)
        r += mItems[i];
    return r;
}

/*
 * Constructor: CMatrix
 * @argument copyFrom: matriz desde donde se quiere realizar al copia..
 * Propósito: Realiza la copia de una matriz como una nueva matriz de tipo CMatrix..
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T>::CMatrix(CMatrix<T> * copyFrom)
{
    mHeight = copyFrom->mHeight;
    mWidth = copyFrom->mWidth;
    mRows = new CArray<T>*[mHeight];
    for(uint i = 0; i < mHeight; i++) {
        mRows[i] = new CArray<T>(mWidth);
        for(uint j = 0; j < mWidth; j++) {
            at(i,j) = copyFrom->at(i,j);
        }
    }
}

/*
 * Constructor: CMatrix
 * @argument height: alto de la imagen como unsigned int.
 * @argument width: ancho de la imagen como unsigned int.
 * Propósito: Realiza la instanciación de un nuevo objeto de tipo CMatrix, de tamaño height x widht.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T>::CMatrix(uint height, uint width)
        : mHeight(height), mWidth(width)
{
    mRows = new CArray<T>*[mHeight];
    for(uint i = 0; i < mHeight; i++)
        mRows[i] = new CArray<T>(mWidth);
}

/*
 * Constructor: CMatrix
 * @argument height: alto de la imagen como unsigned int.
 * @argument width: ancho de la imagen como unsigned int.
 * @argument initialValue: valor inicial que tocará cada elmento de la matriz.
 * Propósito: Realiza la instanciación de un nuevo objeto de tipo CMatrix, de tamaño height x widht en la que todos sus elementos
 * 				tomarán el valor especificado en initialValue.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T>::CMatrix(uint height, uint width, T initialValue)
        : mHeight(height), mWidth(width)
{
    mRows = new CArray<T>*[mHeight];
    for(uint i = 0; i < mHeight; i++) {
        mRows[i] = new CArray<T>(mWidth);
        for(uint j = 0; j < mWidth; j++) {
            at(i,j) = initialValue;
        }
    }
}

/*
 * Constructor: CMatrix
 * @argument im: imagen origen de tipo QImage desde la que se quiere realizar la instanciación del objeto.
 * @argument useR: booleano que indica el uso de la componenete Roja de la imagen.
 * @argument useG: booleano que indica el uso de la componente verde de la imagen.
 * @argument useB: booleano que indica el uso de la componente azul de la imagen.
 * Propósito: Realiza la instanciación de un nuevo objeto de tipo CMatrix, a partir de un tipo de datos QImage de Qt.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T>::CMatrix(QImage * im, bool useR, bool useG, bool useB)
{
    mHeight = im->height();
    mWidth = im->width();

    if(!(useR || useG || useB))
        useR = useG = useB;

    mRows = new CArray<T>*[mHeight];
    for(uint i = 0; i < mHeight; i++) {
        mRows[i] = new CArray<T>(mWidth);
        for(uint j = 0; j < mWidth; j++) {
            uint p = im->pixel(j,i);
            at(i,j) = 0;
            double totalW = 0;
            if(useR) { at(i,j) += (p&0xFF)*0.11; totalW += 0.11; }
            if(useG) { at(i,j) += ((p&0xFF00)>>8)*0.59; totalW += 0.59; }
            if(useB) { at(i,j) += ((p&0xFF0000)>>16)*0.3; totalW += 0.3; }
            at(i,j) /= totalW * 256.;
        }
    }
}

/*
 * Destructor: CMatrix
 * Propósito: Elimina un objeto de tipo CMatrix de memoria.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T>::~CMatrix()
{
    for(uint i = 0; i < mHeight; i++)
        delete mRows[i];
    delete [] mRows;
}

/*
 * Método: operator/=
 * @argument quotient: valor por el que se desea establecer la división de los elementos de la matriz.
 * Propósito: Realiza la división de todos los elementos de la matriz por el valor indicado como arugmento.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> void CMatrix<T>::operator/=(T quotient)
{
    for(uint i = 0; i < mHeight; i++)
        (*mRows[i]) /= quotient;
}

/*
 * Método: operator+=
 * @argument quotient: valor por el que se desea establecer la suma de los elementos de la matriz.
 * Propósito: Realiza la suma de todos los elementos de la matriz por el valor indicado como arugmento.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> void CMatrix<T>::operator+=(CMatrix<T>& summand)
{
    for(uint i = 0; i < mHeight; i++)
        for(uint j = 0; j < mWidth; j++)
            at(i, j) += summand.at(i, j);
}

/*
 * Método: squareElementsInPlace
 * Propósito: Realiza el cuadrado de todos los elementos de la matriz.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> void CMatrix<T>::squareElementsInPlace()
{
    for(uint i = 0; i < mHeight; i++)
        for(uint j = 0; j < mWidth; j++)
            at(i, j) *= at(i, j);
}

/*
 * Método: squareRootElementsInPlace
 * Propósito: Realiza la raíz cuadrada de todos los elementos de la matriz.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> void CMatrix<T>::squareRootElementsInPlace()
{
    for(uint i = 0; i < mHeight; i++)
        for(uint j = 0; j < mWidth; j++)
            at(i, j) = sqrt(at(i, j));
}

/*
 * Método: atan2
 * @argument y: matriz de elementos de tipo CMatrix
 * @argument x: matriz de elementos de tipo CMatrix
 * Propósito: Realiza el cálculo del arcotangente de todos los elementos de la matriz y con los de la matriz x.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T> * CMatrix<T>::atan2(CMatrix<T>& y, CMatrix<T>& x)
{
    CMatrix<T> * out = new CMatrix<T>(y.mHeight, y.mWidth);
    for(uint i = 0; i < y.mHeight; i++)
        for(uint j = 0; j < y.mWidth; j++)
            out->at(i, j) = ::atan2(y[i][j], x[i][j]);
    return out;
}

/*
 * Método: sum
 * Propósito: Realiza el cálculo de la suma de los elementos sobre una matriz de tipo de dato CMAtrix.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> T CMatrix<T>::sum()
{
    T r = 0;    // T debe tener un 0 declarado.
    for(uint i = 0; i < mHeight; i++)
        r += mRows[i]->sum();
    return r;
}

/*
 * Método: filterBy
 * @argument kernel: matriz de tipo CMatrix por la que se quiere realizar la convolución en la matriz que invoca al método.
 * Propósito: Realiza el cálculo de la convolución sobre la matriz que invoca al método, aplicando el filtro argumento.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> CMatrix<T>* CMatrix<T>::filterBy(CMatrix<T>& kernel)
{
    CMatrix<T> * out = new CMatrix<T>(mHeight, mWidth);

    if((kernel.mWidth & 1 == 0) | (kernel.mHeight & 1 == 0)) {
        QMessageBox::critical(0, "Error", "Kernels must have odd size");
        return out;
    }

    int rangeX = (kernel.mHeight-1)/2, rangeY = (kernel.mWidth-1)/2;

    for(uint i = 0; i < mHeight; i++) {
        for(uint j = 0; j < mWidth; j++) {
            out->at(i, j) = 0;
            for(int x = -rangeX; x <= rangeX; x++) {
                int posX = x + i;
                if(posX < 0 || posX >= (int)mHeight) continue;
                for(int y = -rangeY; y <= rangeY; y++) {
                    int posY = y + j;
                    if(posY < 0 || posY >= (int)mWidth) continue;
                    out->at(i, j) += at(posX, posY)*kernel[rangeX+x][rangeY+y];
                }
            }
        }
    }

    return out;
}

/*
 * Método: toNewImage
 * @argument rescale: booleano que indica si se debe realizar un reescalado de la imagen o no.
 * @return QImage: imagen resultado de la copia.
 * Propósito: Realiza el copiado de una matriz de tipo CMatrix a una imagen de tipo QImage.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
template<typename T> QImage * CMatrix<T>::toNewImage(bool rescale)
{
    QImage * out = new QImage(mWidth, mHeight, QImage::Format_RGB32);

    double baseline = 0, scaleFactor = 255.;

    if(rescale) {
        T min = at(0,0), max = at(0,0);
        for(uint i = 0; i < mHeight; i++) {
            for(uint j = 0; j < mWidth; j++) {
                if(min > at(i,j)) min = at(i,j);
                if(max < at(i,j)) max = at(i,j);
            }
        }
        baseline = min;
        if(max != min)
            scaleFactor = 255./(max - min);
        else
            scaleFactor = 255.;
    }
    for(uint i = 0; i < mHeight; i++) {
        for(uint j = 0; j < mWidth; j++) {
            uint c = (uint)ceil((at(i,j)-baseline)*scaleFactor);
            out->setPixel(j, i, qRgb(c,c,c));
        }
    }
    return out;
}

#endif // CMATRIX_H
