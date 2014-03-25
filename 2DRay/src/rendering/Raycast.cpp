#include "rendering/Raycast.h"

#include "maths/Types.h"
#include "util/Util.h"
#include "rendering/Camera.h"

namespace vlr
{
	namespace rendering
	{
		void screenPointToRay(int x, int y, const float4* origin, const mat4* mvp, const viewport* viewport, ray* ray)
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
