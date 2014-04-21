#ifndef VLR_RENDERING_CUDAUTIL
#define VLR_RENDERING_CUDAUTIL

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace vlr
{
	namespace rendering
	{
		#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
		void gpuAssert(cudaError_t code, char *file, int line, bool abort=true);
	}
}

#endif /* VLR_RENDERING_CUDAUTIL */
