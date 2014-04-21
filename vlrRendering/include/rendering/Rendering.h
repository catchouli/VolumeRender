#ifndef VLR_RENDERING_RENDERING
#define VLR_RENDERING_RENDERING

#include <cuda_runtime_api.h>

#include "Camera.h"
#include "Raycast.h"
#include "Octree.h"
#include "OctNode.h"
#include "../maths/Types.h"

namespace vlr
{
	namespace rendering
	{
		// Function declarations
		void renderOctree(const Octree* octreeGpu, const rendering::float4* origin,
			const mat4* mvp, const viewport* viewport);

		__global__ void cudaRenderOctree(const Octree* tree, int* pixel_buffer, const rendering::float4* origin,
			const mat4* mvp, const viewport* viewport);
		
		// Inline definitions
		__device__ __host__ inline void screenPointToRay(int x, int y,
			const float4* origin, const mat4* mvp, const viewport* viewport, ray* ray)
		{
			float4 viewportPos;

			memcpy(&ray->origin, origin, sizeof(float4));

			float width = (float)viewport->w;
			float height = (float)viewport->h;
			float oneOverWidth = 1.0f / width;
			float oneOverHeight = 1.0f / height;

			float normx = x * oneOverWidth;
			float normy = y * oneOverHeight;

			viewportPos.x = normx * 2.0f - 1.0f;
			viewportPos.y = normy * 2.0f - 1.0f;
			viewportPos.z = 1.0f;
			viewportPos.w = 1.0f;

			multMatrixVector(mvp, &viewportPos, &ray->direction);
		}
	}
}

#endif /* VLR_RENDERING_RENDERING */
