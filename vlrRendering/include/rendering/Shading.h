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

		__device__ int shade(const rendering_attributes_t rendering_attributes, glm::vec3 view_dir,
			float hit_t, glm::vec3 hit_pos,
			const int* hit_parent, int hit_idx, int hit_scale);
	}
}

#endif /* VLR_RENDERING_SHADING */
