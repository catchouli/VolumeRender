#include "rendering/Raycast.h"

#ifdef VLR_RAYCAST_CPU

#include "app/Framebuffer.h"
#include "maths/Matrix.h"
#include "maths/Types.h"
#include "util/Util.h"

namespace vlr
{
	namespace rendering
	{
		void raycastOctree(int* p, const ray* ray);

		void renderOctree(const Octree* octree, const float4* origin,
			const mat4* mvp, const viewport* viewport)
		{
			const int width = viewport->w;
			const int height = viewport->h;

			// Initialise framebuffer
			static rendering::Framebuffer fb;
			fb.resize(width, height);

			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					int* pixel = fb.getPointer() + y * width + x;

					// Calculate ray
					ray ray;
					screenPointToRay(x, y, origin, mvp, viewport, &ray);

					// Do raycast
					raycastOctree(pixel, &ray);
				}
			}

			fb.render();
		}

		void raycastOctree(int* p, const ray* ray)
		{
			const int s_max = 23;
			const float epsilon = 0.00001f;

			float4 o = ray->origin;
			float4 d = ray->direction;

			float cubePos[3] = { 0.5f, 0.5f, 0.5f };
			float cubeScale = 1.0f;
		
			float cubeMin[3] = {
				cubePos[0] - 0.5f * cubeScale,
				cubePos[1] - 0.5f * cubeScale,
				cubePos[2] - 0.5f * cubeScale
			};
		
			float cubeMax[3] = {
				cubePos[0] + 0.5f * cubeScale,
				cubePos[1] + 0.5f * cubeScale,
				cubePos[2] + 0.5f * cubeScale
			};

			float cubeCentre[3] = {
				0.5f * (cubeMin[0] + cubeMax[0]),
				0.5f * (cubeMin[1] + cubeMax[1]),
				0.5f * (cubeMin[2] + cubeMax[2]),
			};
		
			if (fabs(d.x) < epsilon) d.x = (d.x < 0.0f ? -epsilon : epsilon);
			if (fabs(d.y) < epsilon) d.y = (d.y < 0.0f ? -epsilon : epsilon);
			if (fabs(d.z) < epsilon) d.z = (d.z < 0.0f ? -epsilon : epsilon);

			float tx_coef = 1.0f / -(d.x);
			float ty_coef = 1.0f / -(d.y);
			float tz_coef = 1.0f / -(d.z);
		
			float tx_bias = tx_coef * o.x;
			float ty_bias = ty_coef * o.y;
			float tz_bias = tz_coef * o.z;

			float txmin = cubeMin[0] * tx_coef + tx_bias;
			float tymin = cubeMin[1] * ty_coef + ty_bias;
			float tzmin = cubeMin[2] * tz_coef + tz_bias;
		
			float txmax = cubeMax[0] * tx_coef + tx_bias;
			float tymax = cubeMax[1] * ty_coef + ty_bias;
			float tzmax = cubeMax[2] * tz_coef + tz_bias;
		
			if (txmin > txmax)
				swap(txmin, txmax);
			if (tymin > tymax)
				swap(tymin, tymax);
			if (tzmin > tzmax)
				swap(tzmin, tzmax);

			float tmin = max(txmin, tymin, tzmin);
			float tmax = min(txmax, tymax, tzmax);
		
			float tcx = cubeCentre[0] * tx_coef + tx_bias;
			float tcy = cubeCentre[1] * ty_coef + ty_bias;
			float tcz = cubeCentre[2] * tz_coef + tz_bias;
		
			bool r = (tcx < tmin) == (d.x >= 0.0f);
			bool g = (tcy < tmin) == (d.y >= 0.0f);
			bool b = (tcz < tmin) == (d.z >= 0.0f);
			//bool g = tcy < tmin;
			//bool b = tcz < tmin;
		
			//if (d.x < 0.0f) r = !r;
			//if (d.y < 0.0f) g = !g;
			//if (d.z < 0.0f) b = !b;

			colour col;
			col.col = -1;
		
			if (r) col.r = 0;
			if (g) col.g = 0;
			if (b) col.b = 0;

			*p = 0;

			if (tmin < tmax)
			{
				*p = col.col;
			}
		}
	}
}

#endif /* VLR_RAYCAST_CPU */
