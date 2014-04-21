#include "util/CudaUtil.h"

namespace vlr
{
	namespace rendering
	{
		void gpuAssert(cudaError_t code, char *file, int line, bool abort)
		{
			if (code != cudaSuccess) 
			{
				fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

				system("pause");

				if (abort)
					exit(code);
			}
		}
	}
}
