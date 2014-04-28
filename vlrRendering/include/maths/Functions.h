#ifndef VLR_RENDERING_MATHS_FUNCTIONS_H
#define VLR_RENDERING_MATHS_FUNCTIONS_H

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

namespace vlr
{
	namespace rendering
	{
		__host__ __device__ int32_t clamp(int32_t x, int32_t a, int32_t b);
	}
}

#endif /* VLR_RENDERING_MATHS_FUNCTIONS_H */
