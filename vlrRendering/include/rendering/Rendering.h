#ifndef VLR_RENDERING_RENDERING
#define VLR_RENDERING_RENDERING

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

#include "Camera.h"
#include "child_desc.h"
#include "Raycast.h"
#include "rendering_attributes.h"
#include "../maths/Types.h"
#include "../resources/Octree.h"

namespace vlr
{
	namespace rendering
	{
		// Function declarations
		void renderOctree(const int32_t* tree, const rendering_attributes_t rendering_attributes);

		__global__ void cudaRenderOctree(const int32_t* tree, int32_t* pixel_buffer,
			const rendering_attributes_t rendering_attributes);
		
		// Inline definitions
		__device__ __host__ ray screenPointToRay(int32_t x, int32_t y,
			const rendering_attributes_t& rendering_attributes);
	}
}

#endif /* VLR_RENDERING_RENDERING */
