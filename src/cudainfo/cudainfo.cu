/*!
	\file cudainfo.cu
	\brief CUDA information functions.
	\author AG
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#if CUDA_VERSION < 2000
#error CUDA 1.1 is not supported any more! Please use CUDA Toolkit 2.0+ instead.
#endif

#include <qglobal.h>
#include "log.h"
#include "cudainfo.h"

#define CZ_COPY_BUF_SIZE	(16 * (1 << 20))	/*!< Transfer buffer size. */
#define CZ_COPY_LOOPS_NUM	8			/*!< Number of loops to run transfer test to. */

#define CZ_CALC_BLOCK_LOOPS	16			/*!< Number of loops to run calculation loop. */
#define CZ_CALC_BLOCK_SIZE	256			/*!< Size of instruction block. */
#define CZ_CALC_BLOCK_NUM	16			/*!< Number of instruction blocks in loop. */
#define CZ_CALC_OPS_NUM		2			/*!< Number of operations per one loop. */
#define CZ_CALC_LOOPS_NUM	8			/*!< Number of loops to run performance test to. */

#define CZ_DEF_WARP_SIZE	32			/*!< Default warp size value. */
#define CZ_DEF_THREADS_MAX	512			/*!< Default max threads value value. */

/*!
	\brief Error handling of CUDA RT calls.
*/
#define CZ_CUDA_CALL(funcCall, errProc) \
	{ \
		cudaError_t errCode; \
		if((errCode = (funcCall)) != cudaSuccess) { \
			CZLog(CZLogLevelError, "CUDA Error: %s", cudaGetErrorString(errCode)); \
			errProc; \
		} \
	}

/*!
	\brief Prototype of function \a cuDeviceGetAttribute().
*/
typedef CUresult (CUDAAPI *cuDeviceGetAttribute_t)(int *pi, CUdevice_attribute attrib, CUdevice dev);

/*!
	\brief Prototype of function \a cuInit().
*/
typedef CUresult (CUDAAPI *cuInit_t)(unsigned int Flags);

/*!
	\brief Pointer to function \a cuDeviceGetAttribute().
	This parameter is initializaed by CZCudaIsInit().
*/
static cuDeviceGetAttribute_t p_cuDeviceGetAttribute = NULL;

/*!
	\brief Pointer to function \a cuInit().
	This parameter is initializaed by CZCudaIsInit().
*/
static cuInit_t p_cuInit = NULL;

#ifdef Q_OS_WIN
//#include <windows.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WINAPI __stdcall
typedef void *HINSTANCE;
typedef HINSTANCE HMODULE;
typedef const char *LPCSTR;
typedef int (WINAPI *FARPROC)();
__out_opt HMODULE WINAPI LoadLibraryA(__in LPCSTR lpLibFileName);
FARPROC WINAPI GetProcAddress(__in HMODULE hModule, __in LPCSTR lpProcName);
#ifdef __cplusplus
}
#endif

/*!
	\brief Check if CUDA fully initialized.
	This function loads nvcuda.dll and finds functions \a cuInit()
	and \a cuDeviceGetAttribute().
	\return \a true in case of success, \a false in case of error.
*/
static bool CZCudaIsInit(void) {

	HINSTANCE hDll;

	if((p_cuInit == NULL) || (p_cuDeviceGetAttribute == NULL)) {

		hDll = LoadLibraryA("nvcuda.dll");
		if(hDll == NULL) {
			return false;
		}

		p_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(hDll, "cuDeviceGetAttribute");
		if(p_cuDeviceGetAttribute == NULL) {
			return false;
		}

		p_cuInit = (cuInit_t)GetProcAddress(hDll, "cuInit");
		if(p_cuInit == NULL) {
			return false;
		}
	}

	return true;
}
#elif defined(Q_OS_LINUX)
#include <dlfcn.h>
/*!
	\brief Check if CUDA fully initialized.
	This function loads libcuda.so and finds functions \a cuInit()
	and \a cuDeviceGetAttribute().
	\return \a true in case of success, \a false in case of error.
*/
static bool CZCudaIsInit(void) {

	void *hDll = NULL;

	if((p_cuInit == NULL) || (p_cuDeviceGetAttribute == NULL)) {

		if(hDll == NULL) {
			hDll = dlopen("/usr/lib/libcuda.so", RTLD_LAZY);
		}

		if(hDll == NULL) {
			hDll = dlopen("/usr/lib32/libcuda.so", RTLD_LAZY);
		}

		if(hDll == NULL) {
			hDll = dlopen("/usr/lib64/libcuda.so", RTLD_LAZY);
		}

		if(hDll == NULL) {
			hDll = dlopen("/usr/lib128/libcuda.so", RTLD_LAZY);
		}

		if(hDll == NULL) {
			return false;
		}

		p_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)dlsym(hDll, "cuDeviceGetAttribute");
		if(p_cuDeviceGetAttribute == NULL) {
			return false;
		}

		p_cuInit = (cuInit_t)dlsym(hDll, "cuInit");
		if(p_cuInit == NULL) {
			return false;
		}
	}
	return true;
}
#else//!Q_OS_WIN
#error Function CZCudaIsInit() is not implemented for your platform!
#endif//Q_OS_WIN


/*!
	\brief Check if CUDA is present here.
*/
bool CZCudaCheck(void) {

	if(!CZCudaIsInit())
		return false;

	if(p_cuInit(0) == CUDA_ERROR_NOT_INITIALIZED) {
		return false;
	}

	return true;
}

/*!
	\brief Check how many CUDA-devices are present.
	\return number of CUDA-devices in case of success, \a 0 if no CUDA-devies were found.
*/
int CZCudaDeviceFound(void) {

	int count;

	CZ_CUDA_CALL(cudaGetDeviceCount(&count),
		return 0);

	return count;
}

/*!
	\brief Read information about a CUDA-device.
	\return \a 0 in case of success, \a -1 in case of error.
*/
int CZCudaReadDeviceInfo(
	struct CZDeviceInfo *info,	/*!< CUDA-device information. */
	int num				/*!< Number (index) of CUDA-device. */
) {
	cudaDeviceProp prop;

	if(info == NULL)
		return -1;

	if(!CZCudaIsInit())
		return -1;

	if(num >= CZCudaDeviceFound())
		return -1;

	CZ_CUDA_CALL(cudaGetDeviceProperties(&prop, num),
		return -1);

	info->num = num;
	strcpy(info->deviceName, prop.name);
	info->major = prop.major;
	info->minor = prop.minor;

	info->core.regsPerBlock = prop.regsPerBlock;
	info->core.SIMDWidth = prop.warpSize;
	info->core.maxThreadsPerBlock = prop.maxThreadsPerBlock;
	info->core.maxThreadsDim[0] = prop.maxThreadsDim[0];
	info->core.maxThreadsDim[1] = prop.maxThreadsDim[1];
	info->core.maxThreadsDim[2] = prop.maxThreadsDim[2];
	info->core.maxGridSize[0] = prop.maxGridSize[0];
	info->core.maxGridSize[1] = prop.maxGridSize[1];
	info->core.maxGridSize[2] = prop.maxGridSize[2];
	info->core.clockRate = prop.clockRate;
	info->core.muliProcCount = prop.multiProcessorCount;
	info->core.watchdogEnabled = prop.kernelExecTimeoutEnabled;

	info->mem.totalGlobal = prop.totalGlobalMem;
	info->mem.sharedPerBlock = prop.sharedMemPerBlock;
	info->mem.maxPitch = prop.memPitch;
	info->mem.totalConst = prop.totalConstMem;
	info->mem.textureAlignment = prop.textureAlignment;
	info->mem.gpuOverlap = prop.deviceOverlap;

	return 0;
}

/*!
	\brief Local service data structure for bandwith calulations.
*/
struct CZDeviceInfoBandLocalData {
	void		*memHostPage;	/*!< Pageable host memory. */
	void		*memHostPin;	/*!< Pinned host memory. */
	void		*memDevice1;	/*!< Device memory buffer 1. */
	void		*memDevice2;	/*!< Device memory buffer 2. */
};

/*!
	\brief Set device for current thread.
*/
int CZCudaCalcDeviceSelect(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	CZLog(CZLogLevelLow, "Selecting %s.", info->deviceName);

	CZ_CUDA_CALL(cudaSetDevice(info->num),
		return -1);

	return 0;
}

/*!
	\brief Allocate buffers for bandwidth calculations.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static int CZCudaCalcDeviceBandwidthAlloc(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {
	CZDeviceInfoBandLocalData *lData;

	if(info == NULL)
		return -1;

	if(info->band.localData == NULL) {

		CZLog(CZLogLevelLow, "Alloc local buffers for %s.", info->deviceName);

		lData = (CZDeviceInfoBandLocalData*)malloc(sizeof(*lData));
		if(lData == NULL) {
			return -1;
		}

		CZLog(CZLogLevelLow, "Alloc host pageable for %s.", info->deviceName);

		lData->memHostPage = (void*)malloc(CZ_COPY_BUF_SIZE);
		if(lData->memHostPage == NULL) {
			free(lData);
			return -1;
		}

		CZLog(CZLogLevelLow, "Host pageable is at 0x%08X.", lData->memHostPage);

		CZLog(CZLogLevelLow, "Alloc host pinned for %s.", info->deviceName);

		CZ_CUDA_CALL(cudaMallocHost((void**)&lData->memHostPin, CZ_COPY_BUF_SIZE),
			free(lData->memHostPage);
			free(lData);
			return -1);

		CZLog(CZLogLevelLow, "Host pinned is at 0x%08X.", lData->memHostPin);

		CZLog(CZLogLevelLow, "Alloc device buffer 1 for %s.", info->deviceName);

		CZ_CUDA_CALL(cudaMalloc((void**)&lData->memDevice1, CZ_COPY_BUF_SIZE),
			cudaFreeHost(lData->memHostPin);
			free(lData->memHostPage);
			free(lData);
			return -1);

		CZLog(CZLogLevelLow, "Device buffer 1 is at 0x%08X.", lData->memDevice1);

		CZLog(CZLogLevelLow, "Alloc device buffer 2 for %s.", info->deviceName);

		CZ_CUDA_CALL(cudaMalloc((void**)&lData->memDevice2, CZ_COPY_BUF_SIZE),
			cudaFree(lData->memDevice1);
			cudaFreeHost(lData->memHostPin);
			free(lData->memHostPage);
			free(lData);
			return -1);

		CZLog(CZLogLevelLow, "Device buffer 2 is at 0x%08X.", lData->memDevice2);

		info->band.localData = (void*)lData;
	}

	return 0;
}

/*!
	\brief Free buffers for bandwidth calculations.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static int CZCudaCalcDeviceBandwidthFree(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {
	CZDeviceInfoBandLocalData *lData;

	if(info == NULL)
		return -1;

	lData = (CZDeviceInfoBandLocalData*)info->band.localData;
	if(lData != NULL) {

		CZLog(CZLogLevelLow, "Free host pageable for %s.", info->deviceName);

		if(lData->memHostPage != NULL)
			free(lData->memHostPage);

		CZLog(CZLogLevelLow, "Free host pinned for %s.", info->deviceName);

		if(lData->memHostPin != NULL)
			cudaFreeHost(lData->memHostPin);

		CZLog(CZLogLevelLow, "Free device buffer 1 for %s.", info->deviceName);

		if(lData->memDevice1 != NULL)
			cudaFree(lData->memDevice1);

		CZLog(CZLogLevelLow, "Free device buffer 2 for %s.", info->deviceName);

		if(lData->memDevice2 != NULL)
			cudaFree(lData->memDevice2);

		CZLog(CZLogLevelLow, "Free local buffers for %s.", info->deviceName);

		free(lData);
	}
	info->band.localData = NULL;

	return 0;
}

/*!
	\brief Reset results of bandwidth calculations.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static int CZCudaCalcDeviceBandwidthReset(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	if(info == NULL)
		return -1;

	info->band.copyHDPage = 0;
	info->band.copyHDPin = 0;
	info->band.copyDHPage = 0;
	info->band.copyDHPin = 0;
	info->band.copyDD = 0;

	return 0;
}

#define CZ_COPY_MODE_H2D	0	/*!< Host to device data copy mode. */
#define CZ_COPY_MODE_D2H	1	/*!< Device to host data copy mode. */
#define CZ_COPY_MODE_D2D	2	/*!< Device to device data copy mode. */

/*!
	\brief Run data transfer bandwidth tests.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static float CZCudaCalcDeviceBandwidthTestCommon (
	struct CZDeviceInfo *info,	/*!< CUDA-device information. */
	int mode,			/*!< Run bandwidth test in one of modes. */
	int pinned			/*!< Use pinned \a (=1) memory buffer instead of pagable \a (=0). */
) {
	CZDeviceInfoBandLocalData *lData;
	float timeMs = 0.0;
	float bandwidthKBs = 0.0;
	cudaEvent_t start;
	cudaEvent_t stop;
	void *memHost;
	void *memDevice1;
	void *memDevice2;
	int i;

	if(info == NULL)
		return 0;

	CZ_CUDA_CALL(cudaEventCreate(&start),
		return 0);

	CZ_CUDA_CALL(cudaEventCreate(&stop),
		cudaEventDestroy(start);
		return 0);

	lData = (CZDeviceInfoBandLocalData*)info->band.localData;

	memHost = pinned? lData->memHostPin: lData->memHostPage;
	memDevice1 = lData->memDevice1;
	memDevice2 = lData->memDevice2;

	CZLog(CZLogLevelLow, "Starting %s test (%s) on %s.",
		(mode == CZ_COPY_MODE_H2D)? "host to device":
		(mode == CZ_COPY_MODE_D2H)? "device to host":
		(mode == CZ_COPY_MODE_D2D)? "device to device": "unknown",
		pinned? "pinned": "pageable",
		info->deviceName);

	for(i = 0; i < CZ_COPY_LOOPS_NUM; i++) {

		float loopMs = 0.0;

		CZ_CUDA_CALL(cudaEventRecord(start, 0),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		switch(mode) {
		case CZ_COPY_MODE_H2D:
			CZ_CUDA_CALL(cudaMemcpy(memDevice1, memHost, CZ_COPY_BUF_SIZE, cudaMemcpyHostToDevice),
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				return 0);
			break;

		case CZ_COPY_MODE_D2H:
			CZ_CUDA_CALL(cudaMemcpy(memDevice2, memHost, CZ_COPY_BUF_SIZE, cudaMemcpyHostToDevice),
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				return 0);
			break;

		case CZ_COPY_MODE_D2D:
			CZ_CUDA_CALL(cudaMemcpy(memDevice2, memDevice1, CZ_COPY_BUF_SIZE, cudaMemcpyDeviceToDevice),
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				return 0);
			break;

		default: // WTF!
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0;
		}

		CZ_CUDA_CALL(cudaEventRecord(stop, 0),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		CZ_CUDA_CALL(cudaEventSynchronize(stop),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		CZ_CUDA_CALL(cudaEventElapsedTime(&loopMs, start, stop),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		timeMs += loopMs;
	}

	CZLog(CZLogLevelLow, "Test complete in %f ms.", timeMs);

	bandwidthKBs = (
		1000 *
		(float)CZ_COPY_BUF_SIZE *
		(float)CZ_COPY_LOOPS_NUM
	) / (
		timeMs *
		(float)(1 << 10)
	);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return (int)bandwidthKBs;
}

/*!
	\brief Run several bandwidth tests.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static int CZCudaCalcDeviceBandwidthTest(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	info->band.copyHDPage = CZCudaCalcDeviceBandwidthTestCommon(info, CZ_COPY_MODE_H2D, 0);
	info->band.copyHDPin = CZCudaCalcDeviceBandwidthTestCommon(info, CZ_COPY_MODE_H2D, 1);
	info->band.copyDHPage = CZCudaCalcDeviceBandwidthTestCommon(info, CZ_COPY_MODE_D2H, 0);
	info->band.copyDHPin = CZCudaCalcDeviceBandwidthTestCommon(info, CZ_COPY_MODE_D2H, 1);
	info->band.copyDD = CZCudaCalcDeviceBandwidthTestCommon(info, CZ_COPY_MODE_D2D, 0);

	return 0;
}

/*!
	\brief Prepare buffers bandwidth tests.
	\return \a 0 in case of success, \a -1 in case of error.
*/
int CZCudaPrepareDevice(
	struct CZDeviceInfo *info
) {

	if(info == NULL)
		return -1;

	if(!CZCudaIsInit())
		return -1;

	if(CZCudaCalcDeviceBandwidthAlloc(info) != 0)
		return -1;

	return 0;
}

/*!
	\brief Calculate bandwidth information about CUDA-device.
	\return \a 0 in case of success, \a -1 in case of error.
*/
int CZCudaCalcDeviceBandwidth(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	if(info == NULL)
		return -1;

	if(CZCudaCalcDeviceBandwidthReset(info) != 0)
		return -1;

	if(!CZCudaIsInit())
		return -1;

	if(CZCudaCalcDeviceBandwidthAlloc(info) != 0)
		return -1;

	if(CZCudaCalcDeviceBandwidthTest(info) != 0)
		return -1;

	return 0;
}

/*!
	\brief Cleanup after test and bandwidth calculations.
	\return \a 0 in case of success, \a -1 in case of error.
*/
int CZCudaCleanDevice(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	if(info == NULL)
		return -1;

	if(CZCudaCalcDeviceBandwidthFree(info) != 0)
		return -1;

	return 0;
}

/*!
	\brief Reset results of preformance calculations.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static int CZCudaCalcDevicePerformanceReset(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	if(info == NULL)
		return -1;

	info->perf.calcFloat = 0;
	info->perf.calcDouble = 0;
	info->perf.calcInteger32 = 0;
	info->perf.calcInteger24 = 0;

	return 0;
}

/*!
	\brief 16 MAD instructions for float point test.
*/
#define CZ_CALC_FMAD_16(a, b) \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \

/*!
	\brief 256 MAD instructions for float point test.
*/
#define CZ_CALC_FMAD_256(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \
	CZ_CALC_FMAD_16(a, b) CZ_CALC_FMAD_16(a, b) \

/*!
	\brief 16 DMAD instructions for double-precision test.
*/
#define CZ_CALC_DFMAD_16(a, b) \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \

/*	a = fma(b, a, b); b = fma(a, b, a); a = fma(b, a, b); b = fma(a, b, a); \
	a = fma(b, a, b); b = fma(a, b, a); a = fma(b, a, b); b = fma(a, b, a); \
	a = fma(b, a, b); b = fma(a, b, a); a = fma(b, a, b); b = fma(a, b, a); \
	a = fma(b, a, b); b = fma(a, b, a); a = fma(b, a, b); b = fma(a, b, a); \*/

/*!
	\brief 256 MAD instructions for float point test.
*/
#define CZ_CALC_DFMAD_256(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \
	CZ_CALC_DFMAD_16(a, b) CZ_CALC_DFMAD_16(a, b) \

/*!
	\brief 16 MAD instructions for 32-bit integer test.
*/
#define CZ_CALC_IMAD32_16(a, b) \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
	a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \

/*!
	\brief 256 MAD instructions for 32-bit integer test.
*/
#define CZ_CALC_IMAD32_256(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \
	CZ_CALC_IMAD32_16(a, b) CZ_CALC_IMAD32_16(a, b) \

/*!
	\brief 16 MAD instructions for 24-bit integer test.
*/
#define CZ_CALC_IMAD24_16(a, b) \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \
	a = __umul24(b, a) + b; b = __umul24(a, b) + a; \

/*!
	\brief 256 MAD instructions for 24-bit integer test.
*/
#define CZ_CALC_IMAD24_256(a, b) \
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\
	CZ_CALC_IMAD24_16(a, b) CZ_CALC_IMAD24_16(a, b)\

#define CZ_CALC_MODE_FLOAT	0	/*!< Single-precision float point test mode. */
#define CZ_CALC_MODE_DOUBLE	1	/*!< Double-precision float point test mode. */
#define CZ_CALC_MODE_INTEGER32	2	/*!< 32-bit integer test mode. */
#define CZ_CALC_MODE_INTEGER24	3	/*!< 24-bit integer test mode. */

/*!
	\brief GPU code for float point test.
*/
static __global__ void CZCudaCalcKernelFloat(void *buf) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float *arr = (float*)buf;
	float val1 = index;
	float val2 = arr[index];
	int i;

	for(i = 0; i < CZ_CALC_BLOCK_LOOPS; i++) {
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
		CZ_CALC_FMAD_256(val1, val2);
	}

	arr[index] = val1 + val2;
}

/*!
	\brief GPU code for double-precision test.
*/
static __global__ void CZCudaCalcKernelDouble(double *buf) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	double *arr = (double*)buf;
	double val1 = index;
	double val2 = arr[index];
	int i;

	for(i = 0; i < CZ_CALC_BLOCK_LOOPS; i++) {
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
		CZ_CALC_DFMAD_256(val1, val2);
	}

	arr[index] = val1 + val2;
}

/*!
	\brief GPU code for 32-bit integer test.
*/
static __global__ void CZCudaCalcKernelInteger32(void *buf) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int *arr = (int*)buf;
	int val1 = index;
	int val2 = arr[index];
	int i;

	for(i = 0; i < CZ_CALC_BLOCK_LOOPS; i++) {
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
		CZ_CALC_IMAD32_256(val1, val2);
	}

	arr[index] = val1 + val2;
}

/*!
	\brief GPU code for 24-bit integer test.
*/
static __global__ void CZCudaCalcKernelInteger24(void *buf) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int *arr = (int*)buf;
	int val1 = index;
	int val2 = arr[index];
	int i;

	for(i = 0; i < CZ_CALC_BLOCK_LOOPS; i++) {
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
		CZ_CALC_IMAD24_256(val1, val2);
	}

	arr[index] = val1 + val2;
}

/*!
	\brief Run GPU calculation performace tests.
	\return \a 0 in case of success, \a -1 in case of error.
*/
static float CZCudaCalcDevicePerformanceTest(
	struct CZDeviceInfo *info,	/*!< CUDA-device information. */
	int mode			/*!< Run performance test in one of modes. */
) {
	CZDeviceInfoBandLocalData *lData;
	float timeMs = 0.0;
	float performanceKOPs = 0.0;
	cudaEvent_t start;
	cudaEvent_t stop;
	int i;

	if(info == NULL)
		return 0;

	CZ_CUDA_CALL(cudaEventCreate(&start),
		return 0);

	CZ_CUDA_CALL(cudaEventCreate(&stop),
		cudaEventDestroy(start);
		return 0);

	lData = (CZDeviceInfoBandLocalData*)info->band.localData;

	int threadsNum = info->core.maxThreadsPerBlock;
	if(threadsNum == 0) {
		int warpSize = info->core.SIMDWidth;
		if(warpSize == 0)
			warpSize = CZ_DEF_WARP_SIZE;
		threadsNum = warpSize * 2;
		if(threadsNum > CZ_DEF_THREADS_MAX)
			threadsNum = CZ_DEF_THREADS_MAX;
	}

	CZLog(CZLogLevelLow, "Starting %s test on %s (%d loops).",
		(mode == CZ_CALC_MODE_FLOAT)? "single-precision float":
		(mode == CZ_CALC_MODE_DOUBLE)? "double-precision float":
		(mode == CZ_CALC_MODE_INTEGER32)? "32-bit integer":
		(mode == CZ_CALC_MODE_INTEGER24)? "24-bit integer": "unknown",
		info->deviceName,
		threadsNum);

	for(i = 0; i < CZ_CALC_LOOPS_NUM; i++) {

		float loopMs = 0.0;

		CZ_CUDA_CALL(cudaEventRecord(start, 0),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		switch(mode) {
		case CZ_CALC_MODE_FLOAT:
			CZCudaCalcKernelFloat<<<1, threadsNum>>>(lData->memDevice1);
			break;

		case CZ_CALC_MODE_DOUBLE:
			CZCudaCalcKernelDouble<<<1, threadsNum>>>((double*)lData->memDevice1);
			break;

		case CZ_CALC_MODE_INTEGER32:
			CZCudaCalcKernelInteger32<<<1, threadsNum>>>(lData->memDevice1);
			break;

		case CZ_CALC_MODE_INTEGER24:
			CZCudaCalcKernelInteger24<<<1, threadsNum>>>(lData->memDevice1);
			break;

		default: // WTF!
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0;
		}

		CZ_CUDA_CALL(cudaGetLastError(),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		CZ_CUDA_CALL(cudaEventRecord(stop, 0),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		CZ_CUDA_CALL(cudaEventSynchronize(stop),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		CZ_CUDA_CALL(cudaEventElapsedTime(&loopMs, start, stop),
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return 0);

		timeMs += loopMs;
	}

	CZLog(CZLogLevelLow, "Test complete in %f ms.", timeMs);

	performanceKOPs = (
		(float)info->core.muliProcCount *
		(float)CZ_CALC_LOOPS_NUM *
		(float)threadsNum *
		(float)CZ_CALC_BLOCK_LOOPS *
		(float)CZ_CALC_OPS_NUM *
		(float)CZ_CALC_BLOCK_SIZE *
		(float)CZ_CALC_BLOCK_NUM
	) / (float)timeMs;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return (int)performanceKOPs;
}

/*!
	\brief Calculate performance information about CUDA-device.
	\return \a 0 in case of success, \a -1 in case of error.
*/
int CZCudaCalcDevicePerformance(
	struct CZDeviceInfo *info	/*!< CUDA-device information. */
) {

	if(info == NULL)
		return -1;

	if(CZCudaCalcDevicePerformanceReset(info) != 0)
		return -1;

	if(!CZCudaIsInit())
		return -1;

	info->perf.calcFloat = CZCudaCalcDevicePerformanceTest(info, CZ_CALC_MODE_FLOAT);
	if(((info->major > 1)) ||
		((info->major == 1) && (info->minor >= 3)))
		info->perf.calcDouble = CZCudaCalcDevicePerformanceTest(info, CZ_CALC_MODE_DOUBLE);
	info->perf.calcInteger32 = CZCudaCalcDevicePerformanceTest(info, CZ_CALC_MODE_INTEGER32);
	info->perf.calcInteger24 = CZCudaCalcDevicePerformanceTest(info, CZ_CALC_MODE_INTEGER24);

	return 0;
}
