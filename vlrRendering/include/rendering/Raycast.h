#ifndef VLR_RAYCAST
#define VLR_RAYCAST

#include <cuda_runtime_api.h>

#include "maths/Matrix.h"
#include "maths/Types.h"
#include "rendering/Camera.h"
#include "rendering/Octree.h"

namespace vlr
{
	namespace rendering
	{
		// Number of threads per block in each dimension (must be a multiple of 32)
		// This is the square of pixels each block fills
		const int THREADS_PER_BLOCK_SQUARE = 32;

		// Maximum scale (depth of octree) (the number of bits in a single precision float)
		const int MAX_SCALE = 23;
		
		// Function Declarations
		Octree* uploadOctreeCuda(const Octree& octree);

		__device__ void raycast(const Octree* tree, const rendering::ray* ray, float* out_hit_t,
			float3* out_hit_pos, OctNode** out_hit_parent, int* out_hit_idx, int* out_hit_scale);
	}
}

#endif /* VLR_RAYCAST */
