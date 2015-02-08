/*
 * FloatMatrix.cpp
 * Propósito: Clase que encapsula una matriz de floats, se empleará para almacenar
 *  		  los filtros de los algoritmos.
 *
 * Creado: 21/03/2012
 * @author: jose
 */

#include "FloatMatrix.h"

FloatMatrix::FloatMatrix() {
	// TODO Auto-generated destructor stub
}

FloatMatrix::~FloatMatrix() {
	// TODO Auto-generated destructor stub
}

FloatMatrix::FloatMatrix(float widht, float height) {
	//Se reserva memoria para la matriz.
	this->elements = (float*)malloc(height*widht*sizeof(float));
	this->setWidth(widht);
	this->setHeight(height);
}

FloatMatrix::FloatMatrix(QImage image) {
}

//Métodos para las operaciones matrix
void FloatMatrix::insertPos(int x,int y,float floatValue){
	//al elemento [x][y] le corresponde el x*M+y siendo M el ancho de la matriz(numero de columnas)
	this->elements[x*this->getWidth()+y] = floatValue;
}

float FloatMatrix::getPos(int x, int y){
	//al elemento [x][y] le corresponde el x*M+y siendo M el ancho de la matriz(numero de columnas)
	float returnValue = this->elements[x*this->getWidth()+y];
	return returnValue;
}


//Implementaciones para getters

/**
 * Devuelve el ancho de la matrix
 * @return int
 */

int FloatMatrix::getWidth() {
	return this->widht;
}

/**
 * Devuelve el alto de la matriz
 * @return int
 */
int FloatMatrix::getHeight() {
	return this->height;
}

/**
 * Devuelve el numero de elementos de la matriz
 * (ancho x alto)
 * @return int
 */
int FloatMatrix::getElementsNumber() {
		return widht*height;
}

/**
 * Obtiene el puntero a float de los elementos de la matriz
 * @return float
 */
float *FloatMatrix::getElements() {
	return this->elements;
}

//Implementaciones para setters

/**
 * Setea el ancho de la matriz
 * @return void
 */
void FloatMatrix::setWidth(float width) {
	this->widht=width;
}

/**
 * Setea el alto de la matriz
 * @return void
 */
void FloatMatrix::setHeight(float height) {
	this->height=height;
}

/**
 * Setea el contenido de la matriz (el puntero a float)
 * @return void
 */
void FloatMatrix::setElements(float *matrix) {
	this->elements=matrix;

}
