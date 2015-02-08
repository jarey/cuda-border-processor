/*
 * PlotWidget.h: definición del componente gráfico encargado de permitir visualizar una gráfica
 * 				 en tres dimensiones para una imagen procesada.
 *
 * Creado: 11/06/2012
 * Autor: jose
 */

#ifndef PLOTWIDGET_H_
#define PLOTWIDGET_H_

#include "./src/ui/graphs/include/qwt3d_surfaceplot.h"
#include "./src/ui/graphs/include/qwt3d_function.h"
#include "./src/ui/graphs/include/qwt3d_plot.h"
#include <QtGui/QWidget>

class PlotWidget : public QWidget{
public:
	//Constructor y destructor.
	PlotWidget();
	~PlotWidget();
	//Atributos públicos para superficie de representación
	//y representación de ejemplo rosernbrock.
	Qwt3D::SurfacePlot* plot;
	Qwt3D::Function *rosenbrock;
	//realización de la representación grafica.
	void doPlot(QImage processedImage);

private:
  int tics;
};

#endif /* PLOTWIDGET_H_ */
