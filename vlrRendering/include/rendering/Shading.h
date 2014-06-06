#ifndef VLR_RENDERING_SHADING
#define VLR_RENDERING_SHADING

#include "child_desc.h"
#include "Raycast.h"
#include "rendering_attributes.h"
#include "../maths/Functions.h"

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

		// Estimate fresnel equation using schlick's approximation
		// http://en.wikipedia.org/wiki/Schlick's_approximation
		__host__ __device__ inline float fresnel(glm::vec3 light_dir, glm::vec3 view_dir, float index1, float index2)
		{
			float r0 = sqr((index1 - index2) / (index1 + index2));

			glm::vec3 half_dir = glm::normalize(-light_dir - view_dir);

			float r = r0 + (1 - r0) * powf((1 - glm::dot(half_dir, view_dir)), 5.0f);

			return r;
		}
		
		template <int recursions>
		__device__ glm::vec4 shade(const rendering_attributes_t rendering_attributes, const ray& ray, RaycastHit& hit,
			StackEntry* stack, const int32_t* tree, float index = 1.0f);
	}
}

#include "Shading.inl"

#endif /* VLR_RENDERING_SHADING */
