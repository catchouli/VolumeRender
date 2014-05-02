#ifndef VLR_RENDERING_CUDAUTIL
#define VLR_RENDERING_CUDAUTIL

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __CUDACC__
	#define HOST_DEVICE_FUNC __host__ __device__
#else
	#define HOST_DEVICE_FUNC
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, char *file, int32_t line, bool abort=true);

namespace vlr
{
	namespace rendering
	{
		namespace test
		{
			template <typename T>
			HOST_DEVICE_FUNC inline void swap(T& a, T& b)
			{
				T temp = a;
				a = b;
				b = temp;
			}
		}
		
		__device__ int32_t get_child_index(uint32_t mask);

		__host__ __device__ int numberOfSetBits(int i);
	}
}

#endif /* VLR_RENDERING_CUDAUTIL */
