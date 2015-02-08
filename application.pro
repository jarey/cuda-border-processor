TEMPLATE = app
QT = gui \
    core \
    network
CONFIG += qt \
    warn_on
CONFIG += release

# CONFIG += debug
# CONFIG += console
CONFIG += static

# message(CONFIG: $$CONFIG)
FORMS = 
HEADERS = src/ui/webcam/VideoWidget.h \
    src/ui/webcam/CaptureThread.h \
    src/ui/timedialog/TimeDialog.h \
    src/ui/Panel6.h \
    src/common/TimeContainer.h \
    src/ui/Panel5.h \
    src/ui/graphs/PlotWidget.h \
    src/ui/graphs/3rdparty/gl2ps/gl2ps.h \
    src/ui/graphs/include/qwt3d_autoptr.h \
    src/ui/graphs/include/qwt3d_autoscaler.h \
    src/ui/graphs/include/qwt3d_axis.h \
    src/ui/graphs/include/qwt3d_color.h \
    src/ui/graphs/include/qwt3d_colorlegend.h \
    src/ui/graphs/include/qwt3d_coordsys.h \
    src/ui/graphs/include/qwt3d_drawable.h \
    src/ui/graphs/include/qwt3d_enrichment.h \
    src/ui/graphs/include/qwt3d_enrichment_std.h \
    src/ui/graphs/include/qwt3d_function.h \
    src/ui/graphs/include/qwt3d_global.h \
    src/ui/graphs/include/qwt3d_graphplot.h \
    src/ui/graphs/include/qwt3d_gridmapping.h \
    src/ui/graphs/include/qwt3d_helper.h \
    src/ui/graphs/include/qwt3d_io.h \
    src/ui/graphs/include/qwt3d_io_gl2ps.h \
    src/ui/graphs/include/qwt3d_io_reader.h \
    src/ui/graphs/include/qwt3d_label.h \
    src/ui/graphs/include/qwt3d_mapping.h \
    src/ui/graphs/include/qwt3d_multiplot.h \
    src/ui/graphs/include/qwt3d_openglhelper.h \
    src/ui/graphs/include/qwt3d_parametricsurface.h \
    src/ui/graphs/include/qwt3d_plot.h \
    src/ui/graphs/include/qwt3d_portability.h \
    src/ui/graphs/include/qwt3d_scale.h \
    src/ui/graphs/include/qwt3d_surfaceplot.h \
    src/ui/graphs/include/qwt3d_types.h \
    src/ui/graphs/include/qwt3d_volumeplot.h \
    src/ui/Panel4.h \
    src/ui/utils/core_global.h \
    src/ui/utils/fancymainwindow.h \
    src/ui/utils/manhattanstyle.h \
    src/ui/utils/minisplitter.h \
    src/ui/utils/styleanimator.h \
    src/ui/utils/styledbar.h \
    src/ui/utils/stylehelper.h \
    src/ui/utils/utils_global.h \
    src/ui/fancytabwidget.h \
    src/cudainfo/build.h \
    src/cudainfo/cudainfo.h \
    src/cudainfo/czdeviceinfo.h \
    src/cudainfo/czdialog.h \
    src/cudainfo/log.h \
    src/cudainfo/version.h \
    src/ui/PanelBase.h \
    src/ui/Panel3.h \
    src/ui/Panel2.h \
    src/cuda/canny/CudaInvoquer.cuh \
    src/cuda/canny/CudaCannyEdgeDetection.cuh \
    src/cuda/canny/CudaZeroCrossing.cuh \
    src/cuda/canny/CudaSobelEdgeDetection.cuh \
    src/cuda/canny/CudaDiscreteGaussian.cuh \
    src/cuda/canny/Cuda2DSeparableConvolution.cuh \
    src/cuda/CudaConvolutionSeparableBusiness.cuh \
    src/cuda/Cuda2dConvolutionBusiness.cuh \
    src/cuda/CudaAlgorythmBusiness.cuh \
    src/cuda/Cuda5StepConvolutionBusiness.cuh \
    src/cuda/Cuda5bStepConvolutionBusiness.cuh \
    src/cuda/CudaRobertCrossShared.cuh \
    src/cuda/CudaLaplaceShared.cuh \
    src/imagefilters/CannyImageFilter.h \
    src/imagefilters/LaplacianOfGaussianImageFilter.h \
    src/imagefilters/LaplaceImageFilter.h \
    src/imagefilters/RobertCrossImageFilter.h \
    src/imagefilters/PrewittImageFilter.h \
    src/imagefilters/SobelSquareImageFilter.h \
    src/imagefilters/ImageFilterFactory.h \
    src/filterexecutors/CannyFilterExecutor.h \
    src/filterexecutors/SobelSquareFilterExecutor.h \
    src/canny/globals.h \
    src/canny/CImage.h \
    src/canny/CMatrix.h \
    src/filterexecutors/LaplacianOfGaussianFilterExecutor.h \
    src/filterexecutors/RobertCrossFilterExecutor.h \
    src/filterexecutors/LaplaceFilterExecutor.h \
    src/filterexecutors/PrewittFilterExecutor.h \
    src/imagefilters/ImageAlgorythmBusiness.h \
    src/imagefilters/ImageProcessingBusiness.h \
    src/common/FloatMatrix.h \
    src/filterexecutors/FilterExecutor.h \
    src/filterexecutors/FilterExecutorFactory.h \
    src/filterexecutors/SobelFilterExecutor.h \
    src/imagefilters/ImageFilter.h \
    src/imagefilters/SobelImageFilter.h \
    src/common/Controlador.h \
    src/common/Constants.h \
    src/ui/droparea.h \
    src/ui/panel1.h \
    src/ui/panelList.h \
    src/ui/mainwindow.h
SOURCES = src/ui/webcam/VideoWidget.cpp \
    src/ui/webcam/CaptureThread.cpp \
    src/ui/timedialog/TimeDialog.cpp \
    src/ui/Panel6.cpp \
    src/common/TimeContainer.cpp \
    src/ui/Panel5.cpp \
    src/ui/graphs/PlotWidget.cpp \
    src/ui/graphs/3rdparty/gl2ps/gl2ps.c \
    src/ui/graphs/src/qwt3d_autoscaler.cpp \
    src/ui/graphs/src/qwt3d_axis.cpp \
    src/ui/graphs/src/qwt3d_color.cpp \
    src/ui/graphs/src/qwt3d_colorlegend.cpp \
    src/ui/graphs/src/qwt3d_coordsys.cpp \
    src/ui/graphs/src/qwt3d_dataviews.cpp \
    src/ui/graphs/src/qwt3d_drawable.cpp \
    src/ui/graphs/src/qwt3d_enrichment_std.cpp \
    src/ui/graphs/src/qwt3d_function.cpp \
    src/ui/graphs/src/qwt3d_gridmapping.cpp \
    src/ui/graphs/src/qwt3d_gridplot.cpp \
    src/ui/graphs/src/qwt3d_io.cpp \
    src/ui/graphs/src/qwt3d_io_gl2ps.cpp \
    src/ui/graphs/src/qwt3d_io_reader.cpp \
    src/ui/graphs/src/qwt3d_label.cpp \
    src/ui/graphs/src/qwt3d_lighting.cpp \
    src/ui/graphs/src/qwt3d_meshplot.cpp \
    src/ui/graphs/src/qwt3d_mousekeyboard.cpp \
    src/ui/graphs/src/qwt3d_movements.cpp \
    src/ui/graphs/src/qwt3d_parametricsurface.cpp \
    src/ui/graphs/src/qwt3d_plot.cpp \
    src/ui/graphs/src/qwt3d_scale.cpp \
    src/ui/graphs/src/qwt3d_surfaceplot.cpp \
    src/ui/graphs/src/qwt3d_types.cpp \
    src/ui/Panel4.cpp \
    src/ui/utils/fancymainwindow.cpp \
    src/ui/utils/manhattanstyle.cpp \
    src/ui/utils/minisplitter.cpp \
    src/ui/utils/styleanimator.cpp \
    src/ui/utils/styledbar.cpp \
    src/ui/utils/stylehelper.cpp \
    src/ui/fancytabwidget.cpp \
    src/cudainfo/czdeviceinfo.cpp \
    src/cudainfo/czdialog.cpp \
    src/cudainfo/log.cpp \
    src/ui/PanelBase.cpp \
    src/ui/Panel3.cpp \
    src/ui/Panel2.cpp \
    src/imagefilters/CannyImageFilter.cpp \
    src/imagefilters/LaplacianOfGaussianImageFilter.cpp \
    src/imagefilters/LaplaceImageFilter.cpp \
    src/imagefilters/RobertCrossImageFilter.cpp \
    src/imagefilters/PrewittImageFilter.cpp \
    src/imagefilters/SobelSquareImageFilter.cpp \
    src/imagefilters/ImageFilterFactory.cpp \
    src/filterexecutors/CannyFilterExecutor.cpp \
    src/filterexecutors/SobelSquareFilterExecutor.cpp \
    src/canny/CImage.cpp \
    src/filterexecutors/LaplacianOfGaussianFilterExecutor.cpp \
    src/filterexecutors/RobertCrossFilterExecutor.cpp \
    src/filterexecutors/LaplaceFilterExecutor.cpp \
    src/filterexecutors/PrewittFilterExecutor.cpp \
    src/imagefilters/ImageAlgorythmBusiness.cpp \
    src/imagefilters/ImageProcessingBusiness.cpp \
    src/main.cpp \
    src/common/FloatMatrix.cpp \
    src/filterexecutors/FilterExecutor.cpp \
    src/filterexecutors/FilterExecutorFactory.cpp \
    src/filterexecutors/SobelFilterExecutor.cpp \
    src/imagefilters/ImageFilter.cpp \
    src/imagefilters/SobelImageFilter.cpp \
    src/common/Controlador.cpp \
    src/common/Constants.cpp \
    src/ui/droparea.cpp \
    src/ui/panel1.cpp \
    src/ui/panelList.cpp \
    src/ui/mainwindow.cpp
RESOURCES = application.qrc
win32:RC_FILE += application.qrc
CUSOURCES = src/cuda/CudaLaplaceShared.cu \
    src/cuda/CudaRobertCrossShared.cu \
    src/cuda/canny/CudaInvoquer.cu \
    src/cuda/canny/CudaCannyEdgeDetection.cu \
    src/cuda/canny/CudaZeroCrossing.cu \
    src/cuda/canny/CudaSobelEdgeDetection.cu \
    src/cuda/canny/CudaDiscreteGaussian.cu \
    src/cuda/canny/Cuda2DSeparableConvolution.cu \
    src/cuda/CudaConvolutionSeparableBusiness.cu \
    src/cuda/Cuda2dConvolutionBusiness.cu \
    src/cuda/CudaAlgorythmBusiness.cu \
    src/cuda/Cuda5StepConvolutionBusiness.cu \
    src/cuda/Cuda5bStepConvolutionBusiness.cu \
    src/cudainfo/cudainfo.cu
CUFLAGS = -gencode \
    arch=compute_10,code=compute_10 \
    -gencode \
    arch=compute_11,code=compute_11 \
    -gencode \
    arch=compute_13,code=compute_13
unix:LIBS += -lcudart \
    -lv4l2 \
    -lqwtplot3d
win32:LIBS += $(CUDA_LIB_PATH)\cuda.lib \
    $(CUDA_LIB_PATH)\cudart.lib

# BUILD_H = src/build.h
# QMAKE_EXTRA_VARIABLES += BUILD_H
# PRE_TARGETDEPS += build_h
# build_h.target = build_h
# build_h.commands = sh ./make_build_svn.sh $(EXPORT_BUILD_H)
# QMAKE_EXTRA_TARGETS += build_h
# QCLEANFILES = \
# Makefile \
# Makefile.Debug \
# Makefile.Release \
# vc80.pdb \
# cuda-z.ncb \
# cudainfo.linkinfo \
# version.nsi \
# build.nsi
# win32:QCLEANFILES += bin\cuda-z.exe
# unix:QCLEANFILES += bin/cuda-z
# QMAKE_EXTRA_VARIABLES += QCLEANFILES
# qclean.target = qclean
# qclean.commands = -$(DEL_FILE) $(EXPORT_QCLEANFILES) #$(EXPORT_BUILD_H)
# qclean.depends = clean
# QMAKE_EXTRA_TARGETS += qclean
win32: { 
    pkg-win32.target = pkg-win32
    pkg-win32.commands = makensis.exe \
        pkg-win32.nsi
    pkg-win32.depends = release
    QMAKE_EXTRA_TARGETS += pkg-win32
}
unix: { 
    pkg-linux.target = pkg-linux
    pkg-linux.commands = sh \
        pkg-linux.sh
    pkg-linux.depends = all
    QMAKE_EXTRA_TARGETS += pkg-linux
}
DESTDIR = bin
OBJECTS_DIR = bld/o
MOC_DIR = bld/moc
UI_DIR = bld/ui
RCC_DIR = bld/rcc

# Cuda extra-compiler for handling files specified in the CUSOURCES variable
win32:QMAKE_CUC = $(CUDA_BIN_PATH)/nvcc.exe
unix:QMAKE_CUC = /usr/local/cuda/bin/nvcc
 { 
    cu.name = Cuda \
        ${QMAKE_FILE_IN}
    cu.input = CUSOURCES
    cu.CONFIG += no_link
    cu.variable_out = OBJECTS
    isEmpty(QMAKE_CUC) { 
        win32:QMAKE_CUC = $(CUDA_BIN_PATH)/nvcc.exe
        else:QMAKE_CUC = nvcc
    }
    isEmpty(CU_DIR):CU_DIR = .
    isEmpty(QMAKE_CPP_MOD_CU):QMAKE_CPP_MOD_CU = 
    isEmpty(QMAKE_EXT_CPP_CU):QMAKE_EXT_CPP_CU = .cu
    win32:INCLUDEPATH += $(CUDA_INC_PATH)
    unix:INCLUDEPATH += /usr/local/cuda/include
    unix:LIBPATH += /usr/local/cuda/lib
    QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS
    DebugBuild:QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_DEBUG
    ReleaseBuild:QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_RELEASE
    QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_RTTI_ON \
        $$QMAKE_CXXFLAGS_WARN_ON \
        $$QMAKE_CXXFLAGS_STL_ON
    
    # message(QMAKE_CUFLAGS: $$QMAKE_CUFLAGS)
    QMAKE_CUEXTRAFLAGS += -Xcompiler \
        $$join(QMAKE_CUFLAGS, ",") \
        $$CUFLAGS
    QMAKE_CUEXTRAFLAGS += $(DEFINES) \
        $(INCPATH) \
        $$join(QMAKE_COMPILER_DEFINES, " -D", -D)
    QMAKE_CUEXTRAFLAGS += -c \
        -G \
        -g
    
    # QMAKE_CUEXTRAFLAGS += -keep
    # QMAKE_CUEXTRAFLAGS += -clean
    QMAKE_EXTRA_VARIABLES += QMAKE_CUEXTRAFLAGS
    cu.commands = $$QMAKE_CUC \
        $(EXPORT_QMAKE_CUEXTRAFLAGS) \
        -o \
        $$OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} \
        ${QMAKE_FILE_NAME}$$escape_expand(\n\t)
    cu.output = $$OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
    silent:cu.commands = @echo \
        nvcc \
        ${QMAKE_FILE_IN} \
        && \
        $$cu.commands
    QMAKE_EXTRA_COMPILERS += cu
    build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
    else:cuclean.CONFIG += recursive
    QMAKE_EXTRA_TARGETS += cuclean
}
