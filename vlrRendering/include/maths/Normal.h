#ifndef VLR_RENDERING_NORMAL_H
#define VLR_RENDERING_NORMAL_H

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

namespace vlr
{
	namespace rendering
	{
		// Helper struct
		struct CubicNormal
		{
			int u : 15;
			int v : 14;
			unsigned int sign : 1;
			unsigned int axis : 2;
		};
		
		__host__ __device__ int clamp(int x, int a, int b);
		__host__ __device__ uint32_t compressNormal(glm::vec3 normal);
		__host__ __device__ glm::vec3 decompressNormal(uint32_t normal);
	}
}

#endif /* VLR_RENDERING_NORMAL_H */
