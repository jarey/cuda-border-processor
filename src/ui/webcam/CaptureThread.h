/*
 * CaptureThread.h
 *
 *  Creado: 25/07/2012
 *  Autor: jose
 *  Descripción: clase encargada de la definición de las propiedades del hilo diferenciado que tomará y procesará las imágenes de cámara web.
 */

#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H

#include <QThread>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include "libv4l2.h"
#include "libv4lconvert.h"
#include "VideoWidget.h"


#define CLEAR(x) memset(&(x), 0, sizeof(x))

class CaptureThread : public QThread
{
public:
    explicit CaptureThread(QWidget *parent = 0);
    ~CaptureThread();
    QImage manageBorderDetection(QImage imageIn);
    bool devam;
    //Widget padre del hilo, nuestro widget de visualización de imágenes de cámara web, incrustado
    //en la ventana de la aplicación.
    VideoWidget *parent;
    struct v4l2_format              fmt;
    struct v4l2_buffer              buf;
    struct v4l2_requestbuffers      req;
    enum v4l2_buf_type              type;
    fd_set                          fds;
    struct timeval                  tv;
    int                             r, fd;
    unsigned int                    n_buffers;
    char                            *dev_name;
    char                            out_name[256];
    FILE                            *fout;
    QWidget	*windowPanel;

    //Declaración de la estructura buffer para recogida de las imágenes.
    struct buffer {
            void   *start;
            size_t length;
    };

    //Definición e implementación de método para apertura del componente V4l2 (camara web).
    static void xioctl(int fh, int request, void *arg)
    {
            int r;

            do {
                    r = v4l2_ioctl(fh, request, arg);
            } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));

            if (r == -1) {
                    fprintf(stderr, "error %d, %s\n", errno, strerror(errno));
                    return;
            }
    };
    void run();
    struct buffer *buffers;
    //Método para detención de la recogida y procesamiento de las imágenes de cámara web.
    void stopUlan();
    //Método para dar comienzo a la recogida y procesamiento de las imágenes de cámara web.
    void startUlan();
};

#endif // CAPTURETHREAD_H
