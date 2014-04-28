#ifndef VLR_RENDERING_SHADING
#define VLR_RENDERING_SHADING

#include "rendering_attributes.h"

#include <glm/glm.hpp>
#include <cuda_runtime_api.h>

namespace vlr
{
	namespace rendering
	{
		struct col
		{
			unsigned char r;
			unsigned char g;
			unsigned char b;
			unsigned char a;
		};

		__device__ int32_t shade(const rendering_attributes_t rendering_attributes,
			float hit_t, glm::vec3 hit_pos, const int32_t* root,
			const int32_t* hit_parent, int32_t hit_idx, int32_t hit_scale);
	}
}

#endif /* VLR_RENDERING_SHADING */
