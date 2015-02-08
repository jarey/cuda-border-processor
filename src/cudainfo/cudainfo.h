/*!
	\file cudainfo.h
	\brief CUDA information data and function definition.
	\author AG
*/

#ifndef CZ_CUDAINFO_H
#define CZ_CUDAINFO_H

#ifdef __cplusplus
extern "C" {
#endif

/*!
	\brief Information about CUDA-device core.
*/
struct CZDeviceInfoCore {
	int		regsPerBlock;		/*!< Total number of registers available per block. */
	int		SIMDWidth;		/*!< Warp size. */
	int		maxThreadsPerBlock;	/*!< Maximum number of threads per block. */
	int		maxThreadsDim[3];	/*!< Maximum sizes of each dimension of a block. */
	int		maxGridSize[3];		/*!< Maximum sizes of each dimension of a grid. */
	int		clockRate;		/*!< Clock frequency in kilohertz. */
	int		muliProcCount;		/*!< Number of mutiprocessors in GPU. */
	int		watchdogEnabled;	/*!< Has run time limit for kernels executed. */
};

/*!
	\brief Information about CUDA-device memory.
*/
struct CZDeviceInfoMem {
	int		totalGlobal;		/*!< Total amount of global memory available on the device in bytes. */
	int		sharedPerBlock;		/*!< Total amount of shared memory available per block in bytes. */
	int		maxPitch;		/*!< Maximum pitch allowed by the memory copy functions that involve memory region allocated through cudaMallocPitch()/cuMemAllocPitch() */
	int		totalConst;		/*!< Total amount of constant memory available on the device in bytes. */
	int		textureAlignment;	/*!< Texture base addresses that are aligned to textureAlignment bytes do not need an offset applied to texture fetches. */
	int		gpuOverlap;		/*!< 1 if the device can concurrently copy memory between host and device while executing a kernel, or 0 if not. */
};

/*!
	\brief Information about CUDA-device bandwidth.
*/
struct CZDeviceInfoBand {
	float		copyHDPage;		/*!< Copy rate from host pageable to device memory in KB/s. */
	float		copyHDPin;		/*!< Copy rate from host pinned to device memory in KB/s. */
	float		copyDHPage;		/*!< Copy rate from device to host pageable memory in KB/s. */
	float		copyDHPin;		/*!< Copy rate from device to host pinned memory in KB/s. */
	float		copyDD;			/*!< Copy rate from device to device memory in KB/s. */
	/* Service part of structure. */
	void		*localData;
};

/*!
	\brief Information about CUDA-device performance.
*/
struct CZDeviceInfoPerf {
	float		calcFloat;		/*!< Single-precision float point calculations performance in KFOPS. */
	float		calcDouble;		/*!< Double-precision float point calculations performance in KFOPS. */
	float		calcInteger32;		/*!< 32-bit integer calculations performance in KOPS. */
	float		calcInteger24;		/*!< 24-bit integer calculations performance in KOPS. */
};

/*!
	\brief Information about CUDA-device.
*/
struct CZDeviceInfo {
	int		num;			/*!< Device index */
	char		deviceName[256];	/*!< ASCII string identifying the device. */
	int		major;			/*!< Major revision numbers defining the device's compute capability. */
	int		minor;			/*!< Minor revision numbers defining the device's compute capability. */
	struct CZDeviceInfoCore	core;
	struct CZDeviceInfoMem	mem;
	struct CZDeviceInfoBand	band;
	struct CZDeviceInfoPerf	perf;
};

bool CZCudaCheck(void);
int CZCudaDeviceFound(void);
int CZCudaReadDeviceInfo(struct CZDeviceInfo *info, int num);
int CZCudaCalcDeviceSelect(struct CZDeviceInfo *info);
int CZCudaPrepareDevice(struct CZDeviceInfo *info);
int CZCudaCalcDeviceBandwidth(struct CZDeviceInfo *info);
int CZCudaCalcDevicePerformance(struct CZDeviceInfo *info);
int CZCudaCleanDevice(struct CZDeviceInfo *info);

#ifdef __cplusplus
}
#endif

#endif//CZ_CUDAINFO_H
