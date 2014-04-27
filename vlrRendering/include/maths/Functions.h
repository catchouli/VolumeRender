#ifndef VLR_RENDERING_MATHS_FUNCTIONS_H
#define VLR_RENDERING_MATHS_FUNCTIONS_H

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

namespace vlr
{
	namespace rendering
	{
		__host__ __device__ int clamp(int x, int a, int b);
	}
}

#endif /* VLR_RENDERING_MATHS_FUNCTIONS_H */
