/*
 * droparea.cpp: Clase encargada de realizar la gestión de los eventos de drag&drop de imágenes
 * desde el escritorio a la aplicación. Cuando el usuario suelta una imagen en el área de la ventana
 * ocupada por este componente, si la imagene s un fichero válido se realiza la carga de la imagen
 * en el panel.
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Clase de gestión de eventos de drag&drop.
 */
 #include <QtGui>
 #include "droparea.h"

/*
 * Constructor: DropArea
 * Propósito: Crea y realiza la configuración inicial del componente.
 *
 * Creado: 12/03/2012
 * Autor: jose
 *
 */
 DropArea::DropArea(QWidget *parent)
     : QGraphicsView(parent)
 {
     setMinimumSize(200, 200);
     setFrameStyle(QFrame::Sunken | QFrame::StyledPanel);
     setAlignment(Qt::AlignCenter);
     setAcceptDrops(true);
     setAutoFillBackground(true);
     clear();
 }

 /*
  * Método: dragEnterEvent
  * Propósito: Gestionan el evento de entrada de la acción drag&drop.
  *
  * Creado: 12/03/2012
  * Autor: jose
  *
  */
 void DropArea::dragEnterEvent(QDragEnterEvent *event)
 {
	 //Se pone el fondo oscuro
     setBackgroundRole(QPalette::Highlight);
     //Se llama al método de resolver acción.
     event->acceptProposedAction();
 }

 /*
  * Método: dragMoveEvent
  * Propósito: Detecta el movimiento del ratón con un fichero arrastrado.
  *
  * Creado: 12/03/2012
  * Autor: jose
  *
  */
 void DropArea::dragMoveEvent(QDragMoveEvent *event)
 {
	 //Se llama al método de resolver acción.
     event->acceptProposedAction();
 }

 /*
  * Método: dropEvent
  * Propósito: Se define el evento de soltado de la imagen sobre el area del panel.
  * Se gestiona si al soltar se reconocen urls de imagen en el dato mime para realizar
  * la apertura de la imagen en el panel
  *
  * Creado: 12/03/2012
  * Autor: jose
  *
  */
 void DropArea::dropEvent(QDropEvent *event)
 {
     const QMimeData *mimeData = event->mimeData();

     if (mimeData->hasImage()) {
         //no se hace nada, incluido por escalabilidad del código
     } else if (mimeData->hasHtml()) {
    	 //no se hace nada, incluido por escalabilidad del código
     } else if (mimeData->hasText()) {
    	 //no se hace nada, incluido por escalabilidad del código
     } else if (mimeData->hasUrls()) {
    	 //Se obtienen las rutas de las imágenes
         QList<QUrl> urlList = mimeData->urls();
         QString text;
         for (int i = 0; i < urlList.size() && i < 32; ++i) {
             QString url = urlList.at(i).path();
             text += url + QString("\n");
         }
     } else {
    	 //caso en el que no se encuentre ninguna coincidencia con formatos esperados.
     }
     //Se informa el fondo a oscuro y se llama al método de aceptar la acción.
     setBackgroundRole(QPalette::Dark);
     event->acceptProposedAction();
     emit changed(event->mimeData());
 }

 /*
  * Método: dragLeaveEvent
  * Propósito: Gestiona el evento de la salida de la acción drag&drop del area del panel.
  *
  * Creado: 12/03/2012
  * Autor: jose
  *
  */
 void DropArea::dragLeaveEvent(QDragLeaveEvent *event)
 {
     clear();
     event->accept();
 }

 /*
  * Método: clear
  * Propósito: Realiza la señalización de un cambio en el sistema de eventos.
  *
  * Creado: 12/03/2012
  * Autor: jose
  *
  */
 void DropArea::clear()
 {
     emit changed();
 }
