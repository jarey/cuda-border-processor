/*
 * CImage.cpp
 *
 *  Creado: 20/04/2012
 *  Autor: jose
 *  Descripción: Clase de implementación de la definición establecida en CImage.h.
 *  			 Clase que encapsula una imagen de tipo QImage de QT para aportarle
 *  			 funcionalidad a mayores que se empleará en el proceso de Canny.
 *  			 Además este objeto contiene toda la funcionalidad relacionada con el
 *  			 proceso de canny:
 *  			 - Creación de filtro de Gauss para difuminado de la imagen.
 *  			 - Procesado de filtro de Gauss sobre la imagen.
 *  			 - Aplicación de filtro de Prewitt sobre la imagen.
 *  			 - Supresión de no-máximos sobre la imagen.
 *  			 - Aplicación de histéresis sobre la imagen.
 *  			 - Proceso global de canny empleando todas las características previamente
 *  			 	enumeradas.
 */

#include "CImage.h"

/*
 * Constructor: CImage
 * @argument w: Ancho de la imagen a construir.
 * @argument h: Alto de la imagen a construir.
 * Propósito: Realiza la creación de una nueva imagen para el proceso de Canny con el alto
 * 			  y el ancho especificados. Además establece la propiedad de supresión de no-maximos
 * 			  como no llevada a cabo. La imagen se establece en formato RGB de 32 bits.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */

CImage::CImage(uint w, uint h)
        : mWidth(w), mHeight(h)
{
    mSuppressed = 0;
    mImage = new QImage(w, h, QImage::Format_RGB32);
    mOriginalImage = new QImage(w, h, QImage::Format_RGB32);
}

/*
 * Constructor: CImage
 * @argument file: Ruta del fichero de imagen desde donde se quiere proceder a
 * 				   instanciar una nueva imagen.
 * Propósito: Realiza la creación de una nueva imagen para el proceso de Canny a partir
 * 			 de un fichero de imagen en disco. Ademaś establece la propiedad
 * 			 de supresión de no-maximos como no llevada a cabo.
 * 			 La imagen se establece en formato RGB de 32 bits.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */

CImage::CImage(QString file)
{
    mSuppressed = 0;
    mImage = new QImage(file);
    if(mImage->isNull()){
        QMessageBox::critical(0, "Error", "No se pudo realizar la carga del fichero.");
    }
    mOriginalImage = new QImage(file);
}

/*
 * Constructor: CImage
 * @argument image: Objeto de tipo QImage que se desea emplear para instanciar una imagen
 * 					para el proceso de canny.
 * Propósito: Realiza la creación de una nueva imagen para el proceso de Canny a partir
 * 			 de un objeto de tipo QImage de Qt. Ademaś establece la propiedad
 * 			 de supresión de no-maximos como no llevada a cabo.
 * 			 La imagen se establece en formato RGB de 32 bits.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */

CImage::CImage(QImage image)
{

    mSuppressed = 0;
    mImage = new QImage(image);
    if(mImage->isNull()){
        QMessageBox::critical(0, "Error", "No se puedo intanciar el objeto.");
    }
    mOriginalImage = new QImage(image);
}

/*
 * Destructor: CImage
 * Propósito: Realiza la liberación de memoria para los elementos del objeto de imagen para
 * 			  el proceso de canny.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
CImage::~CImage()
{
    delete mImage;
    delete mOriginalImage;
    if(mSuppressed != 0){ delete mSuppressed;}
}

/*
 * Método: useSuppressed
 * Propósito: Si existe la imagen de supresión de no-máximos en el objeto distinta de nulo
 * 			  entonces sobreescribe el atributo de la imagen de trabajo con dicha imagen
 * 			  para mostrar dicha imagen pro pantalla.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
void CImage::useSuppressed()
{
    if(mSuppressed == 0){
    	return;
    }
    delete mImage;
    mImage = mSuppressed->toNewImage();
}

/*
 * Método: useHysteresis
 * @argument thresholdLow: umbral inferior para emplear en el proceso de histéresis.
 * @argumento thresholdHigh: umbral superior para emplear en el proceso de histéresis.
 * Propósito: Aplica histéresis sobre la imagen de supresión de no-máximos si esta existe y,
 * 			  tras aplicar dicho proceso sobreescribe la imagen de trabajo para mostrar por
 * 			  pantalla la imagen procesada con histeresis.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
void CImage::useHysteresis(double thresholdLow, double thresholdHigh)
{
    if(mSuppressed == 0) return;
    CMatrix<int> * traced = hysteresis(*mSuppressed, thresholdLow, thresholdHigh);

    delete mImage;
    //Se asigna sobre la imagen de trabajo la imagen procesada con histéresis.
    mImage = traced->toNewImage();
    //Se elimina la imagen temporal.
    delete traced;
}

/*
 * Método: canny
 * @argument blurSigma: umbral inferior para emplear en el proceso de histéresis.
 * @argumento useR: booleano que indica si emplear la componente en Rojo de la imagen.
 * @argumento useG: booleano que indica si emplear la componente en Verde de la imagen.
 * @argumento useB: booleano que indica si emplear la componente en Azul de la imagen.
 *
 * Propósito: Ejecuta los pasos del proceso de canny hasta el punto de ejecutar la supresión
 * 			  de no-máximos sobre la imagen. La histéresis, paso siguiente del algoritmo,
 * 			  como en la aplicación lo marcarmos con un flag que puede ser o no marcado,
 * 			  y por lo tanto dicho paso será ejecutado según el valor de ese componente de
 * 			  pantalla, será ejecutado en un proceso distinto. El algoritmo por lo tanto
 * 			  ejecuta:
 * 			  - Crea el filtro de gauss para difuminar la imagen.
 * 			  - Aplica el filtro de gauss sobre la imagen apra difuminarla.
 * 			  - Aplica el filtrado a través de Prewitt sobre la imagen para determinar gradiente.
 * 			  - Realiza la supresión de no-máximos sobre la imagen procesada.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
void CImage::canny(double blurSigma, bool useR, bool useG, bool useB)
{
	//1- Se crea el filtro de gauss para difuminado.
    CMatD gaussian = gaussianFilter(blurSigma);

    //Se instancia la imagen de trabajo.
    CMatD image(mImage, useR, useG, useB);
    mHeight = image.mHeight;
    mWidth = image.mWidth;

    //Se realiza el filtrado de la imagen mediante el filtro de gauss para difuminarla.
    CMatD * filtered = image.filterBy(gaussian);

    //Se establecen los valores del filtro de prewitt en X e Y.
    CMatD prewittX(3, 3, 0), prewittY(3, 3, 0);
    for(uint i = 0; i < 3; i++) {
        prewittX[i][0] = -1;
        prewittX[i][2] = +1;
        prewittY[0][i] = -1;
        prewittY[2][i] = +1;
    }

    //Se realiza el filtrado de la imagen mediante el filtro de Prewitt (en X e Y).
    CMatD * gradX = filtered->filterBy(prewittX);
    CMatD * gradY = filtered->filterBy(prewittY);
    //Se establecen los valores de los gradientes en variables temporales.
    CMatD * grad = new CMatD(gradX);
    CMatD * temp = new CMatD(gradY);

    //Se calcula el cuadrado de los gradientes locales calculado en X e Y.
    grad->squareElementsInPlace();
    temp->squareElementsInPlace();
    grad->operator +=(*temp);
    //Se calcula el gradiente final en cada punto de la imagen.
    grad->squareRootElementsInPlace();
    CMatD * theta = CMatD::atan2(*gradY, *gradX);
    //Se declara la variable local para cálculo de ángulos.
    CMatrix<int> * thetaClamped = new CMatrix<int>(mHeight, mWidth);
    //Para cada ángulo posible se establece un valor:
    //0-45º:3
    //45-90º:2
    //90-135º:1
    //135-180º:4
    for(uint i = 0; i < mHeight; i++)
        for(uint j = 0; j < mWidth; j++) {
            double t = theta->at(i,j);
            if(t < 0) t += M_PI;

            if(t <= M_PI/8) thetaClamped->at(i,j) = 3;
            else if(t <= 3*M_PI/8) thetaClamped->at(i,j) = 2;
            else if(t <= 5*M_PI/8) thetaClamped->at(i,j) = 1;
            else if(t <= 7*M_PI/8) thetaClamped->at(i,j) = 4;
            else thetaClamped->at(i,j) = 3;
        }

    //Si ya existe una matriz de supresión de no-maximos previa, se elimina.
    if(mSuppressed != 0){
    	delete mSuppressed;
    }

    //Se ejecuta la supresión de no-máximos sobre la imagen.
    mSuppressed = suppression(*grad, *thetaClamped);

    //Se libera la memoria para la imagen de trabajo y se le asigna la imagen procesada.
    delete mImage;
    mImage = image.toNewImage();

    //Se eliminan los objetos locales empleados.
    delete theta;
    delete grad;
    delete temp;
    delete gradX;
    delete gradY;
    delete filtered;
    delete thetaClamped;
}


/*
 * Método: suppression
 * @argument grad: matriz de aproximación del gradiente en cada punto de la imagen origen.
 * @argument theta: matriz de ángulo de la dirección del gradiente en cada punto de la imagen.
 * @return CMatD: matriz con la supresión de no-máximos re
 * Propósito: Ejecuta el proceso de supresión de no-máximos sobre la imagen procesada.
 * 			  Se recorre la imagen de gradiente en su alto y su ancho y para cada elemento se
 * 			  calculan sus píxeles adyacentes según la dirección del gradiente en el punto.
 * 			  Posteriormente se calcula si el gradiente en el punto es superior al gradiente de
 * 			  sus vecinos permanece en la matriz, en caso contrario se elimina de la misma informando
 * 			  un valor de gradiente 0.
 *
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
CMatD * CImage::suppression(CMatD& grad, CMatrix<int>& theta)
{
    CMatD * out = new CMatD(&grad);
    //Se recorre la matriz y para cada ángulo se obtienen los "vecinos" correspondientes.
    for(uint i = 0; i < mHeight; i++)
        for(uint j = 0; j < mWidth; j++) {
            int ax, ay, bx, by;
            int angle = theta[i][j];
            if(angle == 1) {
                ax = i+1; ay = j;
                bx = i-1; by = j;
            } else if(angle == 2) {
                ax = i+1; ay = j+1;
                bx = i-1; by = j-1;
            } else if(angle == 3) {
                ax = i; ay = j+1;
                bx = i; by = j-1;
            } else if(angle == 4) {
                ax = i-1; ay = j+1;
                bx = i+1; by = j-1;
            } else { qDebug() << angle; QMessageBox::critical(0, "Error", "Corrupt angle."); return out; }

            //Se comprueba si el valor evaluado es un máximo con respecto a sus vecinos y si no lo es
            //se informa un 0.
            if(ax < 0 || ax >= (int)mHeight || ay < 0 || ay >= (int)mWidth) continue;
            else if(grad[ax][ay] > grad[i][j]) { out->at(i,j) = 0; continue; }

            if(bx < 0 || bx >= (int)mHeight || by < 0 || by >= (int)mWidth) continue;
            else if(grad[bx][by] > grad[i][j]) { out->at(i,j) = 0; continue; }
        }
    //Se retorna la imagen procesada con la supresión de no-maximos.
    return out;
}


/*
 * Método: hysteresis
 * @argument grad: matriz de aproximación del gradiente en cada punto de la imagen origen.
 * @argument thresholdLow: umbral de histéresis inferior.
 * @argument thresholdHigh: umbral de histéresis superior.
 * @return CMatrix: imagen procesada mediante histéresis.
 *
 * Propósito: Ejecuta el proceso de histéresis sobre una imagen, filtrando los píxeles según los valores
 * 			  de los umbrales establecidos.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
CMatrix<int> * CImage::hysteresis(CMatD& grad, double thresholdLow, double thresholdHigh)
{
	//Se declara la pila de procesamiento.
    queue< pair<int, int> > nodes;
    //Se declara la matriz de salida.
    CMatrix<int> * out = new CMatrix<int>(mHeight, mWidth, 0);

    //Se recorre la imagen
    for(uint i = 0; i < mHeight; i++){
        for(uint j = 0; j < mWidth; j++) {
        	//Si el valor del pixen en i,j supera el umbral superior se inserta en la pila de procesamiento.
            if((grad[i][j] >= thresholdHigh) && out->at(i,j) != 1) {
                nodes.push(pair<uint, uint>(i,j));
                //Mientras la lista no esté vacía
                while(!nodes.empty()) {
                	//Se obtienen los nodos al principio de la lsita y se procesan.
                    pair<int, int> node = nodes.front();
                    nodes.pop();
                    int x = node.first, y = node.second;
                    //Si el valor obtenido está dentro de la imagen y es superior al umbral inferior.
                    if(x < 0 || x >= (int)mHeight || y < 0 || y >= (int)mWidth) continue;
                    if(grad[x][y] < thresholdLow) continue;
                    if(out->at(x,y) != 1) {
                    	//Se informa como blanco y se insertan todos sus vecinos en la pila.
                        out->at(x,y) = 1;
                        nodes.push(pair<uint, uint>(x+1,y-1));
                        nodes.push(pair<uint, uint>(x+1,y  ));
                        nodes.push(pair<uint, uint>(x+1,y+1));
                        nodes.push(pair<uint, uint>(x  ,y+1));
                        nodes.push(pair<uint, uint>(x  ,y-1));
                        nodes.push(pair<uint, uint>(x-1,y-1));
                        nodes.push(pair<uint, uint>(x-1,y  ));
                        nodes.push(pair<uint, uint>(x-1,y+1));
                    }
                }
            }
        }
}
    //Se retorna la imagen procesada.
    return out;
}

/*
 * Método: gaussianFilter
 * @argument sigma: valor de la campana de gauss para le filtro de Gauss.
 * @return CMatD: matriz del filtro compuesto apra el sigma dado.
 *
 * Propósito: Construye el filtro de gauss a partir de la desviación típoca de la campana de Guass especificada.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
CMatD CImage::gaussianFilter(double sigma)
{
    /* The Gaussian filter is separable, but for simplicity we implement it as a square filter */
    int n = (int)(2 * floor( (float)sqrt(-log(0.1) * 2 * (sigma*sigma)) ) + 1);
    //Calculo el tamaño del filtro a través del valor de la desviación.
    CMatD r(n,n);

    //Construye los valores del filtro en cada punto de acuerdo a la distribución de gauss.
    for(int i = -(n-1)/2; i <= (n-1)/2; i++)
        for(int j = -(n-1)/2; j <= (n-1)/2; j++) {
            r[(n-1)/2+i][(n-1)/2+j] = (float)exp(-((float)((i*i)+(j*j))/(2*(sigma*sigma))));
        }

    //Normalizo los valores del filtro
    r /= r.sum();

    //Devolución del filtro creado.
    return r;
}
