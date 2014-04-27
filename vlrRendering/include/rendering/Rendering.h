#ifndef VLR_RENDERING_RENDERING
#define VLR_RENDERING_RENDERING

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

#include "Camera.h"
#include "Raycast.h"
#include "rendering_attributes.h"
#include "Octree.h"
#include "../maths/Types.h"
#include "child_desc.h"

namespace vlr
{
	namespace rendering
	{
		// Function declarations
		void renderOctree(const int* tree, const rendering_attributes_t rendering_attributes);

		__global__ void cudaRenderOctree(const int* tree, int* pixel_buffer,
			const rendering_attributes_t rendering_attributes);
		
		// Inline definitions
		__device__ __host__ inline ray screenPointToRay(int x, int y,
			const rendering_attributes_t rendering_attributes);
	}
}

#endif /* VLR_RENDERING_RENDERING */
