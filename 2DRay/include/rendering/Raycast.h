#define VLR_RAYCAST_CPU

#ifndef VLR_RAYCAST
#define VLR_RAYCAST

#include "maths/Types.h"
#include "rendering/Camera.h"

namespace vlr
{
	namespace rendering
	{
		void renderOctree(const float4* origin, const mat4* mvp,
			const viewport* viewport);
		void screenPointToRay(int x, int y, const float4* origin, const mat4* mvp,
			const viewport* viewport, ray* ray);
	}
}

#endif /* VLR_RAYCAST */
