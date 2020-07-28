# cuda-border-processor

C++ / Cuda application for border detection in images (pictures and video) applying real time processing.

The application compares CPU vs GPU implementation of a set of algorythms (gauss, laplacian of gauss, prewitt, sobel, canny, and an implementation of a variant of sobel algorythm optimized for low illuminated images called sobel square filter).

The comparison between both algortythm implementations is shown as a performance indication when they're executed.

Also a mode to compare the algorythms between them, based on image characteristics is implemented.

3D plots with the intensity of the gradient in each point of the images are available.

Video demo of the application: https://www.youtube.com/watch?v=J6E2is0xl6o

Tu build the application you would need to install the Cuda Sdk. To run the GPU version of the algorythms a Cuda capable GPU is needed.
