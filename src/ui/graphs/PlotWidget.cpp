/*
 * PlotWidget.cpp: clase encargada de realizar el grafico 3D en el panel, este componente
 * estará asociado al panel de la funcionalidad correspondiente en la aplicación.
 *
 *  Created on: 11/06/2012
 *      Author: jose
 */

#include "PlotWidget.h"
#include "./src/ui/graphs/include/qwt3d_function.h"
#include <QHBoxLayout>
#include <QDebug>

using namespace std;
using namespace Qwt3D;

// Función de graficamiento, se aprte del ejemplo Rosenbrock del conjunto de ejemplos de la
//librería QwtPlot3D y se transforma en la representación gráfica del gradiente de la imagen
//procesada
class Rosenbrock : public Function
{
public:

	Rosenbrock(SurfacePlot& pw,QImage image )
	:Function(pw)
	{
		this->setImage(image);
	}

	double operator()(double x, double y)
	{
		return qGray(image.pixel(x,y));
	}
	void setImage(QImage image){
		this->image=image;
	}
private:
	QImage image;
};

/*
 * Constructor: PlotWidget
 * Propósito: Se crea el componente y el layout en el que se establecerá el mismo.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
PlotWidget::PlotWidget() {

	//Crear los componentes y el layout
	QHBoxLayout *grid = new QHBoxLayout(this);

	plot = new SurfacePlot();
	grid->addWidget( plot);

	this->setLayout(grid);
}

/*
 * Método: doPlot
 * Propósito: Realiza la representación gráfica para una imagen procesada.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
void PlotWidget::doPlot(QImage processedImage){

	//Titulo de la representación
	plot->setTitle("Gráfica De Gradiente");
	//Establecimiento de los datos de representación.
    rosenbrock = new Rosenbrock(*plot,processedImage);
    rosenbrock->setMesh(100,100);
    rosenbrock->setDomain(0,processedImage.width() ,0,processedImage.height());
    rosenbrock->setMinZ(-10);
    //Creación de la representación.
    rosenbrock->create();
    //Configuración de la representación.
    plot->setRotation(30,0,15);
    plot->setScale(1,1,1);
    plot->setShift(0.15,0,0);
    plot->setZoom(0.9);

    for (unsigned i=0; i!=plot->coordinates()->axes.size(); ++i)
    {
    	plot->coordinates()->axes[i].setMajors(7);
    	plot->coordinates()->axes[i].setMinors(4);
    }

    plot->coordinates()->axes[X1].setLabelString("x-axis");
    plot->coordinates()->axes[Y1].setLabelString("y-axis");
    plot->setCoordinateStyle(BOX);

    plot->updateData();
    plot->updateGL();
  }


/*
 * Destructor: PlotWidget
 * Propósito: Realiza la destrucción de los elementos.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
PlotWidget::~PlotWidget() {
}
