#include "maths/Normal.h"

#include <cuda.h>
#include <math_functions.h>

namespace vlr
{
	namespace rendering
	{
		__host__ __device__ int clamp(int x, int a, int b)
		{
			return x < a ? a : (x > b ? b : x);
		}

		__host__ __device__ uint32_t compressNormal(glm::vec3 normal)
		{
			// Calculate absolute values of normal components
			glm::vec3 absNorm(abs(normal.x), abs(normal.y), abs(normal.z));

			// Work out which axis this is (the component with the greatest magnitude)
			int axis = (absNorm.x >= max(absNorm.y, absNorm.z))
				? 0 : (absNorm.y >= absNorm.z) ? 1 : 2;

			// Arrange tuv values
			glm::vec3 tuv;

			switch (axis)
			{
			case 0:
				tuv = normal;
				break;
			case 1:
				tuv = glm::vec3(normal.y, normal.z, normal.x);
				break;
			default:
				tuv = glm::vec3(normal.z, normal.x, normal.y);
				break;
			}

			// Calculate u and v
			float u = tuv.y / fabsf(tuv.x);
			float v = tuv.z / fabsf(tuv.x);

			// Convert u and v to fixed point
			int32_t fixed_u = clamp((int32_t)(u * 16383.0f), -0x4000, 0x3FFF) & 0x7FFF;
			int32_t fixed_v = clamp((int32_t)(v * 8191.0f), -0x2000, 0x1FFF) & 0x3FFF;

			// Compress values
			int encoded_sign_bit = (tuv.x >= 0.0f) ? 0 : 0x80000000;
			int encoded_axis = axis << 29;
			int encoded_u = fixed_u << 14;
			int encoded_v = fixed_v;

			return encoded_sign_bit |
				   encoded_axis |
				   encoded_u |
				   encoded_v;
		}

		__host__ __device__ glm::vec3 decompressNormal(uint32_t normal)
		{
			// Extract values
			int32_t sign = (int32_t)normal >> 31;
			float t = (float)(sign ^ 0x7fffffff);
			float u = (float)((int32_t)normal << 3);
			float v = (float)((int32_t)normal << 18);

			// Create result based on sign bit
			float3 result = { t, u, v };

			// Y bit
			if ((normal & 0x20000000) != 0)
				result.x = v, result.y = t, result.z = u;

			// Z bit
			else if ((normal & 0x40000000) != 0)
				result.x = u, result.y = v, result.z = t;

			// Reinterpret to glm::vec3
			return *(glm::vec3*)&result;
		}
	}
}
