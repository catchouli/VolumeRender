#include "rendering/Shading.h"

#include "rendering/Raycast.h"
#include "maths/Colour.h"
#include "maths/Normal.h"
#include "util/CUDAUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>

namespace vlr
{
	namespace rendering
	{
		const int SHADING_MAX_RECURSION = 4;

		__device__ inline glm::vec3 reflect(glm::vec3 dir, glm::vec3 norm)
		{
			return norm * (2.0f * glm::dot(dir, norm)) - dir;
		}

		__device__ inline glm::vec4 shade(const rendering_attributes_t rendering_attributes, const ray& eye_ray, RaycastHit& hit,
			StackEntry* stack, const int32_t* tree, float index = 1.0f)
		{
			return shade<0>(rendering_attributes, eye_ray, hit, stack, tree, index);
		}
		
		template <>
		__device__ inline glm::vec4 shade<SHADING_MAX_RECURSION>(const rendering_attributes_t rendering_attributes, const ray& eye_ray, RaycastHit& hit,
			StackEntry* stack, const int32_t* tree, float index)
		{
			return glm::vec4(rendering_attributes.ambient_colour, 1.0f);
		}

		__device__ inline const raw_attachment* lookupRawAttachment(const int32_t* root,
													const int32_t* parent, int child_idx)
		{
			const child_desc_word_1* parent_desc = (child_desc_word_1*)parent;

			int32_t hit_parent_offset_bytes = (char*)parent - (char*)root;
			int32_t info_ptr_offset = hit_parent_offset_bytes & ~(0x2000 - 1);
			const int32_t* info_ptr_ptr = (int32_t*)((uintptr_t)root + info_ptr_offset);
			int32_t info_ptr = *info_ptr_ptr;

			// Load info section
			const info_section* info = (info_section*)((uintptr_t)root + info_ptr);

			// Skip to raw lookup table
			const int32_t* raw_lookup = (int32_t*)((uintptr_t)root + info->raw_lookup);

			// Skip to block of lookup pointers
			int raw_lookup_offset = (hit_parent_offset_bytes / child_desc_size);
			raw_lookup += raw_lookup_offset;

			// Get parent's child block
			const raw_attachment* attachment = (raw_attachment*)((uintptr_t)root + *raw_lookup);
			
			// Skip to child's attachment
			int child_offset = get_child_index(parent_desc->child_mask << child_idx) - 1;
			attachment += child_offset;

			return attachment;
		}
	}
}