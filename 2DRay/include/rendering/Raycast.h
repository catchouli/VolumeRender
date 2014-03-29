//#define VLR_RAYCAST_CPU

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
		struct StackEntry
		{
			rendering::OctNode* parent;
			float tmax;
		};

		Octree* uploadOctreeCuda(const Octree& octree);

		void renderOctree(const Octree* octree, const rendering::float4* origin,
			const mat4* mvp, const viewport* viewport);

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

#endif /* VLR_RAYCAST */
