#ifndef VLR_RENDERING_SHADING
#define VLR_RENDERING_SHADING

#include "rendering_attributes.h"
#include "child_desc.h"

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

		__device__ const raw_attachment* lookupRawAttachment(const int32_t* root,
													const int32_t* parent, int child_idx);

		__device__ glm::vec4 shade(const rendering_attributes_t rendering_attributes,
			const float& hit_t, const glm::vec3& hit_pos, const int32_t* root,
			const int32_t* hit_parent, const int32_t& hit_idx, const int32_t& hit_scale,
			const int iteration = 0);
	}
}

#endif /* VLR_RENDERING_SHADING */
