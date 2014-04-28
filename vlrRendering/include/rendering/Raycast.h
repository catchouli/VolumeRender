#ifndef VLR_RAYCAST
#define VLR_RAYCAST

#include <cuda_runtime_api.h>

#include "../maths/Matrix.h"
#include "../maths/Types.h"
#include "Camera.h"
#include "Octree.h"

namespace vlr
{
	namespace rendering
	{
		// Number of threads per block in each dimension (must be a multiple of 32)
		// This is the square of pixels each block fills
		const int32_t THREADS_PER_BLOCK_SQUARE = 32;

		// Maximum scale (depth of octree) (the number of bits in a single precision float)
		const int32_t MAX_SCALE = 23;

		__device__ void raycast(const int32_t* root, const rendering::ray* ray, float* out_hit_t,
			glm::vec3* out_hit_pos, const int32_t** out_hit_parent, int32_t* out_hit_idx, int32_t* out_hit_scale);
	}
}

#endif /* VLR_RAYCAST */
