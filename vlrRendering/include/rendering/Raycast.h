#ifndef VLR_RAYCAST
#define VLR_RAYCAST

#include <cuda_runtime_api.h>

#include "Camera.h"
#include "../maths/Matrix.h"
#include "../maths/Types.h"
#include "../resources/Octree.h"

namespace vlr
{
	namespace rendering
	{
		// Number of threads per block in each dimension (must be a multiple of 32)
		// This is the square of pixels each block fills
		const int32_t THREADS_PER_BLOCK_SQUARE = 32;

		// Maximum scale (depth of octree) (the number of bits in a single precision float)
		const int32_t MAX_SCALE = 23;

		// An entry in the stack
		struct StackEntry
		{
			const int32_t* parent;
			float t_max;
		};

		// The result of a raycast
		struct RaycastHit
		{
			float hit_t;
			glm::vec3 hit_pos;
			const int32_t* hit_parent;
			int32_t hit_idx;
			glm::vec3 hit_pos_internal;
			int32_t hit_scale;
		};

		// Traverses an octree until it finds an existent voxel
		__device__ void raycast(const int32_t* tree, const rendering::ray* ray, StackEntry* stack,
								RaycastHit* raycastHit);

		// Traverses an octree through a solid, until it finds an empty voxel,
		// at which time it returns the most recent existent voxel hit
		__device__ void raycast_empty(const int32_t* tree, const rendering::ray* ray, StackEntry* stack,
								RaycastHit* raycastHit, const RaycastHit* old_hit);
	}
}

#endif /* VLR_RAYCAST */
