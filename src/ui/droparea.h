/*
 * droparea.h: Clase encargada de realizar la gestión de los eventos de drag&drop de imágenes
 * desde el escritorio a la aplicación. Cuando el usuario suelta una imagen en el área de la ventana
 * ocupada por este componente, si la imagene s un fichero válido se realiza la carga de la imagen
 * en el panel.
 *
 *  Creado: 25/03/2011
 *  Autor: jose
 *  Descripción: Clase de gestión de eventos de drag&drop.
 */

 #ifndef DROPAREA_H
 #define DROPAREA_H

 #include <QLabel>
 #include <QGraphicsView>

 class QMimeData;

 class DropArea : public QGraphicsView
 {
     Q_OBJECT

 public:
     DropArea(QWidget *parent = 0);

 public slots:
     void clear();

 signals:
     void changed(const QMimeData *mimeData = 0);

 protected:
     void dragEnterEvent(QDragEnterEvent *event);
     void dragMoveEvent(QDragMoveEvent *event);
     void dragLeaveEvent(QDragLeaveEvent *event);
     void dropEvent(QDropEvent *event);

 private:
     QGraphicsView label;
 };

 #endif
