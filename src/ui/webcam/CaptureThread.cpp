/*
 * CaptureThread.cpp
 *
 *  Creado: 25/07/2012
 *  Autor: jose
 *  Descripción: clase encargada del procesamiento en un hilo diferenciado de la obtención y procesamiento
 *  			de imágenes a través de la cámara web del equipo.
 */

#include "CaptureThread.h"
#include <QApplication>
#include <QDataStream>
#include <QString>
#include <QDebug>
#include <QBuffer>
#include <typeinfo>
#include <iostream>
#include <QFile>
#include <QMessageBox>
#include "./src/ui/Panel6.h"
#include "./src/imagefilters/ImageFilterFactory.h"
#include "./src/filterexecutors/FilterExecutorFactory.h"
#include "./src/imagefilters/ImageProcessingBusiness.h"
#include "./src/common/Constants.h"
#include "./src/common/Controlador.h"


/*
 * Constructor: CaptureThread
 * @argument parent: QWIdget padre de la invocación, será nuestro panel de visualización de las imágenes en la pantalla principal,
 * 					de modo que amntengamos la referencia desde el hilo para poder actualizar las imágenes procesadas.
 * Creado: 25/07/2012
 * Autor: jose
 * Propósito: construir una instancia del hilo para recoger y procesar las imágenes.
 */
CaptureThread::CaptureThread(QWidget *parent) :
	QThread(parent) {
	Panel6 *panel = dynamic_cast<Panel6 *> (parent);
	this->windowPanel = parent;
	this->parent = (VideoWidget*) panel->frameOrig;
	devam = false;
	fd = -1;
}

/*
 * Método: run
 * Creado: 25/07/2012
 * Autor: jose
 * Propósito: realizar la lógica de negocio del hilo de ejecución, se realizará la apertura del dispositivo, la toma de la imagen, el procesamiento de la misma
 * 			  y la actualización de la imagen en el padre en un bucle infinito controlado por la variable fd.
 */
void CaptureThread::run() {
	//Se establece la condición de parada a -1 y la ruta del dispositivo de vídeo en /dev/video0
	fd = -1;
	dev_name = "/dev/video0";

	//apertura del dispositivo v4l2 del distema (camara web).
	fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
	if (fd < 0) {
		qDebug("No se puede realizar la apertura del dispositivo.");
		return;
	}

	static struct v4lconvert_data *v4lconvert_data;
	static struct v4l2_format src_fmt;
	static unsigned char *dst_buf;

	CLEAR(fmt);
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	//Se establece las dimensiones de recogida de las imágenes a 640x480 píxeles.
	fmt.fmt.pix.width = 640;
	fmt.fmt.pix.height = 480;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
	xioctl(fd, VIDIOC_S_FMT, &fmt);
	if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
		printf("Libv4l no aceptó el formado RGB24. No se puede continuar.\n");
		//exit(EXIT_FAILURE);
		return;
	}
	if ((fmt.fmt.pix.width != 640) || (fmt.fmt.pix.height != 480))
		printf("Advertencia: el driver provee la imagen a %dx%d pixeles\n",
				fmt.fmt.pix.width, fmt.fmt.pix.height);

	v4lconvert_data = v4lconvert_create(fd);
	if (v4lconvert_data == NULL)
		qDebug("v4l conversión necesaria");
	if (v4lconvert_try_format(v4lconvert_data, &fmt, &src_fmt) != 0)
		qDebug("v4l try format");
	xioctl(fd, VIDIOC_S_FMT, &src_fmt);
	dst_buf = (unsigned char*) malloc(fmt.fmt.pix.sizeimage);

	CLEAR(req);
	req.count = 2;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
	xioctl(fd, VIDIOC_REQBUFS, &req);

	//Se añade la imagen tomada al buffer.
	buffers = (buffer*) calloc(req.count, sizeof(*buffers));
	for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
		CLEAR(buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = n_buffers;

		xioctl(fd, VIDIOC_QUERYBUF, &buf);

		buffers[n_buffers].length = buf.length;
		buffers[n_buffers].start = v4l2_mmap(NULL, buf.length,
				PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

		if (MAP_FAILED == buffers[n_buffers].start) {
			qDebug("mmap");
			//exit(EXIT_FAILURE);
			return;
		}
	}

	//se incrementa la dimensión del buffer
	for (int i = 0; i < n_buffers; ++i) {
		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;
		xioctl(fd, VIDIOC_QBUF, &buf);
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	xioctl(fd, VIDIOC_STREAMON, &type);

	int di = 0;
	char header[] = "P6\n640 480 255\n";
	while (devam) {
		//Se controla el timeout de la recepción de imágenes desde el dispositivo
		do {
			FD_ZERO(&fds);
			FD_SET(fd, &fds);

			/* Timeout. */
			tv.tv_sec = 2;
			tv.tv_usec = 0;

			r = select(fd + 1, &fds, NULL, NULL, &tv);
		} while ((r == -1 && (errno = EINTR)));
		if (r == -1) {
			qDebug("select");
			//exit(1) ;
			return;
		}

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		xioctl(fd, VIDIOC_DQBUF, &buf);

		try {

			if (v4lconvert_convert(v4lconvert_data, &src_fmt, &fmt,
					(unsigned char*) buffers[buf.index].start, buf.bytesused,
					dst_buf, fmt.fmt.pix.sizeimage) < 0) {
				if (errno != EAGAIN)
					qDebug("v4l_convert");

			}

			unsigned char* asil = (unsigned char*) malloc(
					fmt.fmt.pix.sizeimage + qstrlen(header));
			memmove(asil, dst_buf, fmt.fmt.pix.sizeimage);
			memmove(asil + qstrlen(header), asil, fmt.fmt.pix.sizeimage);
			memcpy(asil, header, qstrlen(header));
			//Se declara una variable de tipo imagen para comentar el procesamiento de las imágenes tomadas.
			QImage qq;

			if (qq.loadFromData(asil, fmt.fmt.pix.sizeimage + qstrlen(header),
					"PPM")) {
				if (parent->isVisible()) {
					QImage q1(qq);

					//Realizar operaciones sobre la imagen Q1
					q1=this->manageBorderDetection(q1);

					//TODO - Cambiar cuando se encuentre solución al problema del driver en el portatil.
					//Se voltea 180º la imagen para verla del derecho
					QMatrix rm;
					rm.rotate(180);
					q1 = q1.transformed(rm);
					//****
					//actualización de la imagen en el panel padre del hilo para ver las imñagenes en tiempo real.
					parent->img = q1;
					parent->update();
				}
				if (asil)
					free(asil);
			}
		} catch (...) {
		}
		xioctl(fd, VIDIOC_QBUF, &buf);
		di++;
	}
	try {
		type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		xioctl(fd, VIDIOC_STREAMOFF, &type);
		for (int i = 0; i < n_buffers; ++i)
			v4l2_munmap(buffers[i].start, buffers[i].length);

		v4l2_close(fd);
	} catch (...) {
	}
}

/*
 * Constructor: CaptureThread
 * Creado: 25/07/2012
 * Autor: jose
 * Propósito: realiza una instancia del hilo inicializando los parámetros de recogida de imñagenes del mismo.
 */
CaptureThread::~CaptureThread() {
	try {
		type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		xioctl(fd, VIDIOC_STREAMOFF, &type);
		v4l2_close(fd);
	} catch (...) {
	}
	fd = -1;
}

/*
 * Método: stopUlan
 * Creado: 25/07/2012
 * Autor: jose
 * Propósito: realiza la parada de recogida de imágenes por el hilo de la cámara web.
 */
void CaptureThread::stopUlan() {
	devam = false;
	try {
		type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		xioctl(fd, VIDIOC_STREAMOFF, &type);
		v4l2_close(fd);
	} catch (...) {
	}
	fd = -1;

}

/*
 * Método: startUlan
 * Creado: 25/07/2012
 * Autor: jose
 * Propósito: realiza el inicio de la recogida de imágenes de la cámara web por el hilo.
 */
void CaptureThread::startUlan() {
	this->start();

}

/*
 * Método: manageBorderDetection
 * @argument imageIn: imagen de entrada de tipo QImage, recogida desde la cámara web para ser procesada.
 * @return QImage: imagen procesada por el algoritmo seleccionado en pantalla.
 * Creado: 25/07/2012
 * Autor: jose
 * Propósito: realiza el procesamiento por el algoritmo seleccionado en pantalla de las imágenes tomadas por la cñamara web.
 */
QImage CaptureThread::manageBorderDetection(QImage imageIn) {
	Panel6 *panel = dynamic_cast<Panel6 *> (this->windowPanel);
	Controlador *controlador = Controlador::Instance();

	controlador->getMatrixListOrigin()->append(imageIn);

	bool histeresis = panel->histeresisCheck->isChecked();
	int radioGauss = panel->radioGauss->value();
	float sigmaGauss = panel->sigmaGauss->value();
	float lowerThreshold = panel->umbralInferior->value();
	float higherThreshold = panel->umbralSuperior->value();

	ImageFilterFactory *fac = new ImageFilterFactory();
	ImageFilter *imageFilter = fac->getImageFilterInstance(
	panel->algorythmSelect->currentText(), histeresis, radioGauss,
	sigmaGauss, lowerThreshold, higherThreshold);

	//Realizar la llamada al business de ejecución
	//Se instancia el business de negocio
	ImageProcessingBusiness *imageProcessingBusiness;
	//Se realiza la llamada pasando el filtro (negocio determinará el tipo de filtro con la información disponible en el controlador.)
	imageProcessingBusiness->doProcess(*imageFilter);

	QImage imageReturned = controlador->getMatrixListDestiny()->at(0);
	//Despues del load realizamos el borrado:
	controlador->cleanResultList();
	controlador->getMatrixListOrigin()->clear();

	//Borrado de memoria de todos los elementos empleados.
	free(imageProcessingBusiness);
	free(fac);
	free(imageFilter);

	return imageReturned;

}
