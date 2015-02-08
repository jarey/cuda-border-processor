#include <stdio.h>
#include <stdlib.h>
//#include <cutil_inline.h>

#include "CudaRobertCrossShared.cuh"

// Texture reference for reading image
texture<unsigned char, 2> texRobertCross;
extern __shared__ unsigned char LocalBlockRobertCross[];
static cudaArray *arrayRobertCross = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif



__device__ unsigned char
ComputeRobertCross(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale )
{
    short Horz = -ml + mr;
    short Vert = -um+ mr;
    short Sum = (short) (fScale*(abs(Horz)+abs(Vert)));
    if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
    return (unsigned char) Sum;
}

__global__ void
RobertCrossShared( uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
             short BlockWidth, short SharedPitch,
#endif
             short w, short h, float fScale )
{
    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
    	LocalBlockRobertCross[SharedIdx+4*ib+0] = tex2D( texRobertCross,
            (float) (u+4*ib-RADIUS+0), (float) (v-RADIUS) );
    	LocalBlockRobertCross[SharedIdx+4*ib+1] = tex2D( texRobertCross,
            (float) (u+4*ib-RADIUS+1), (float) (v-RADIUS) );
    	LocalBlockRobertCross[SharedIdx+4*ib+2] = tex2D( texRobertCross,
            (float) (u+4*ib-RADIUS+2), (float) (v-RADIUS) );
    	LocalBlockRobertCross[SharedIdx+4*ib+3] = tex2D( texRobertCross,
            (float) (u+4*ib-RADIUS+3), (float) (v-RADIUS) );
    }
    if ( threadIdx.y < RADIUS*2 ) {
        //
        // se copian tiras de tamaño RADIUS*2 a memoria comaprtida
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
        for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
        	LocalBlockRobertCross[SharedIdx+4*ib+0] = tex2D( texRobertCross,
                (float) (u+4*ib-RADIUS+0), (float) (v+blockDim.y-RADIUS) );
        	LocalBlockRobertCross[SharedIdx+4*ib+1] = tex2D( texRobertCross,
                (float) (u+4*ib-RADIUS+1), (float) (v+blockDim.y-RADIUS) );
        	LocalBlockRobertCross[SharedIdx+4*ib+2] = tex2D( texRobertCross,
                (float) (u+4*ib-RADIUS+2), (float) (v+blockDim.y-RADIUS) );
        	LocalBlockRobertCross[SharedIdx+4*ib+3] = tex2D( texRobertCross,
                (float) (u+4*ib-RADIUS+3), (float) (v+blockDim.y-RADIUS) );
        }
    }

    __syncthreads();

    u >>= 2;    // los indices son uchar desde este punto
    uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

        unsigned char pix00 = LocalBlockRobertCross[SharedIdx+4*ib+0*SharedPitch+0];
        unsigned char pix01 = LocalBlockRobertCross[SharedIdx+4*ib+0*SharedPitch+1];
        unsigned char pix02 = LocalBlockRobertCross[SharedIdx+4*ib+0*SharedPitch+2];
        unsigned char pix10 = LocalBlockRobertCross[SharedIdx+4*ib+1*SharedPitch+0];
        unsigned char pix11 = LocalBlockRobertCross[SharedIdx+4*ib+1*SharedPitch+1];
        unsigned char pix12 = LocalBlockRobertCross[SharedIdx+4*ib+1*SharedPitch+2];
        unsigned char pix20 = LocalBlockRobertCross[SharedIdx+4*ib+2*SharedPitch+0];
        unsigned char pix21 = LocalBlockRobertCross[SharedIdx+4*ib+2*SharedPitch+1];
        unsigned char pix22 = LocalBlockRobertCross[SharedIdx+4*ib+2*SharedPitch+2];

        uchar4 out;

        out.x = ComputeRobertCross(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale );

        pix00 = LocalBlockRobertCross[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlockRobertCross[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlockRobertCross[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeRobertCross(pix01, pix02, pix00,
                             pix11, pix12, pix10,
                             pix21, pix22, pix20, fScale );

        pix01 = LocalBlockRobertCross[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlockRobertCross[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlockRobertCross[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeRobertCross( pix02, pix00, pix01,
                              pix12, pix10, pix11,
                              pix22, pix20, pix21, fScale );

        pix02 = LocalBlockRobertCross[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlockRobertCross[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlockRobertCross[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeRobertCross( pix00, pix01, pix02,
                              pix10, pix11, pix12,
                              pix20, pix21, pix22, fScale );
        if ( u+ib < w/4 && v < h ) {
            pSobel[u+ib] = out;
        }
    }

    __syncthreads();
}

__global__ void
RobertCopyImage( Pixel *pSobelOriginal, unsigned int Pitch,
                int w, int h, float fscale )
{
    unsigned char *pSobel =
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        pSobel[i] = min( max((tex2D( texRobertCross, (float) i, (float) blockIdx.x ) * fscale), 0.f), 255.f);
    }
}

__global__ void
RobertTex( Pixel *pSobelOriginal, unsigned int Pitch,
          int w, int h, float fScale )
{
    unsigned char *pSobel =
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        unsigned char pix00 = tex2D( texRobertCross, (float) i-1, (float) blockIdx.x-1 );
        unsigned char pix01 = tex2D( texRobertCross, (float) i+0, (float) blockIdx.x-1 );
        unsigned char pix02 = tex2D( texRobertCross, (float) i+1, (float) blockIdx.x-1 );
        unsigned char pix10 = tex2D( texRobertCross, (float) i-1, (float) blockIdx.x+0 );
        unsigned char pix11 = tex2D( texRobertCross, (float) i+0, (float) blockIdx.x+0 );
        unsigned char pix12 = tex2D( texRobertCross, (float) i+1, (float) blockIdx.x+0 );
        unsigned char pix20 = tex2D( texRobertCross, (float) i-1, (float) blockIdx.x+1 );
        unsigned char pix21 = tex2D( texRobertCross, (float) i+0, (float) blockIdx.x+1 );
        unsigned char pix22 = tex2D( texRobertCross, (float) i+1, (float) blockIdx.x+1 );
        pSobel[i] = ComputeRobertCross(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale );
    }
}

extern "C" void setupTextureRobert(int iw, int ih, Pixel *data, int Bpp)
{
    cudaChannelFormatDesc desc;

    if (Bpp == 1) {
        desc = cudaCreateChannelDesc<unsigned char>();
    } else {
        desc = cudaCreateChannelDesc<uchar4>();
    }

    cudaMallocArray(&arrayRobertCross, &desc, iw, ih);
    cudaMemcpyToArray(arrayRobertCross, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice);
}

extern "C" void deleteTextureRobert(void)
{
    cudaFreeArray(arrayRobertCross);
}


extern "C" void robertCrossFilter(Pixel *odata, int iw, int ih, float fScale)
{
    cudaBindTextureToArray(texRobertCross, arrayRobertCross);

        	printf("DENTRO DE LA FUNCION DEL .CU\n");
            dim3 threads(16,4);
#ifndef FIXED_BLOCKWIDTH
	          int BlockWidth = 80; // debe ser divisible entre 16 para aprovechamiento de coalescencia
#endif
        		dim3 blocks = dim3(iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                               ih/threads.y+(0!=ih%threads.y));
        		int SharedPitch = ~0x3f&(4*(BlockWidth+2*RADIUS)+0x3f);
        		int sharedMem = SharedPitch*(threads.y+2*RADIUS);

        		// el ancho debe ser divisible entre 4.
        		iw &= ~3;
        		RobertCrossShared<<<blocks, threads, sharedMem>>>((uchar4 *) odata,
                                                        iw,
#ifndef FIXED_BLOCKWIDTH
                                                        BlockWidth, SharedPitch,
#endif
                                                		    iw, ih, fScale );


    cudaUnbindTexture(texRobertCross);
}


CudaRobertCrossShared::CudaRobertCrossShared() {
	// TODO Auto-generated constructor stub

}

CudaRobertCrossShared::~CudaRobertCrossShared() {
	// TODO Auto-generated destructor stub
}


float* CudaRobertCrossShared::convolve(float *imagen, int ancho,
		int alto, int brillo) {
	Pixel *img;
	Pixel *h_img;
	int DATA_SIZE=0;
	DATA_SIZE = alto * ancho * sizeof(unsigned char);
	cudaMalloc((void **) &img, DATA_SIZE);
	h_img = (unsigned char*) malloc(DATA_SIZE);
	int i=0;
	int x=0;
	for(i=0;i<ancho*alto;i++){
		h_img[i] = (unsigned char)imagen[i];
	}

	setupTextureRobert(ancho,alto,h_img,1);
	robertCrossFilter(img, ancho, alto, brillo );

	//Se copia el resultado de la ejecución de vuelta a CPU
	cudaMemcpy(h_img, img, DATA_SIZE, cudaMemcpyDeviceToHost);

	float *h_result;
	h_result = (float*) malloc( alto * ancho * sizeof(float));
	for(i=0;i<ancho*alto;i++){
		h_result[i] = (float)h_img[i];
		}
	return h_result;
}
