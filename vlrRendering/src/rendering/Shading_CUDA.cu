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
		__device__ glm::vec3 reflect(glm::vec3 dir, glm::vec3 norm)
		{
			//return 2.0f * norm * glm::dot(dir, norm) - dir;
			return norm * (2.0f * glm::dot(dir, norm)) - dir;
		}
		
		__device__ glm::vec4 shade(const rendering_attributes_t rendering_attributes,
			const float& hit_t, const glm::vec3& hit_pos, const int32_t* root,
			const int32_t* hit_parent, const int32_t& hit_idx, const int32_t& hit_scale,
			const int iteration)
		{
			const int MAX_REFLECTION_ITERATIONS = 0;

			child_desc_word_1* hit_parent_word_1 = (child_desc_word_1*)hit_parent;

			// Result
			glm::vec4 out(0, 0, 0, 0);

			// Calculate view direction
			glm::vec3 view_dir = glm::normalize(rendering_attributes.origin - hit_pos);

			// Look up raw attachment
			const raw_attachment* hit_voxel_attachment =
				lookupRawAttachment(root, hit_parent, hit_idx);

			// Load normal
			glm::vec3 normal = glm::normalize(decompressNormal(hit_voxel_attachment->normal));
			//glm::vec3 normal = glm::normalize(decompressNormal(hit_parent[1]));

			// Load colour
			glm::vec3 colour = glm::vec3(decompressColour(hit_voxel_attachment->colour));
			//glm::vec3 colour = glm::vec3(decompressColour(hit_parent[2]));

			// Calculate lighting
			// Calculate ambient
			out += glm::vec4(rendering_attributes.ambient_colour * colour, 0);

			// For each light
			for (int32_t i = 0; i < rendering_attributes.light_count; ++i)
			{
				float light_distance = exp2f(23);

				const light_t& light = rendering_attributes.lights[i];

				// Light direction (light.direction for directional lights)
				glm::vec3 light_dir = glm::normalize(light.direction);

				// Attenuation (1.0f for directional lights)
				float attenuation = 1.0f;

				// Calculate direction and attentuation for non directional lights
				if (light.type != LightTypes::DIRECTIONAL)
				{
					// Calculate light direction
					glm::vec3 light_diff = hit_pos - light.position;
					light_dir = glm::normalize(light_diff);

					// Calculate light distance
					light_distance = glm::length(light_diff);

					// Calculate attenuation
					attenuation = 1.0f /
						(light.constant_att + light.linear_att * light_distance +
						light.quadratic_att * light_distance * light_distance);

					// If this is a spotlight
					if (light.type == LightTypes::SPOT)
					{
						float clampedCosine = fmaxf(0.0f, glm::dot(light_dir, glm::normalize(light.direction)));

						// If this is outside the spotlight cone
						if (clampedCosine < cos(light.cutoff))
						{
							attenuation = 0.0f;
						}
						else
						{
							attenuation = attenuation * pow(clampedCosine, light.exponent);
						}
					}
				}

				// Check if light hits this position
				if (glm::dot(light_dir, normal) < 0)
				{
					const float outset_size = 0.02f;

					ray light_ray;
					light_ray.direction = -light_dir;

					// The origin is outset a little so we don't just hit the same voxel
					light_ray.origin = hit_pos + light_ray.direction * outset_size;

					float hit_t;						// Resultant hit t value from raycast
					glm::vec3 hit_pos;					// Resultant hit position from raycast
					const int32_t* hit_parent;				// Resultant hit parent voxel from raycast
					int32_t hit_idx;						// Resultant hit child index from raycast
					int32_t hit_scale;						// Resultant hit scale from raycast

					// Do raycast
					raycast(root, &light_ray, &hit_t, &hit_pos, &hit_parent, &hit_idx, &hit_scale, true);

					// If we hit a voxel in the tree
					if (hit_scale < MAX_SCALE && hit_t < light_distance)
					{
						continue;
					}
				}

				// Calculate diffuse reflection
				float diffuse_factor = attenuation * fmaxf(0.0f, glm::dot(normal, -light_dir));
				glm::vec3 diffuse_colour = colour * light.diffuse;

				out += diffuse_factor *  glm::vec4(diffuse_colour, 0.0f);

				// Calculate specular reflection
				float specular_factor = 0.0f;
				glm::vec3 specular_colour = colour * light.specular;

				float specular_exp = 32.0f;

				// If the normal faces the light
				if (glm::dot(normal, light_dir) < 0.0f)
				{
					// Calculate reflection ray
					glm::vec3 reflection = glm::reflect(light_dir, normal);

					specular_factor = powf(fmaxf(0.0f, glm::dot(reflection, view_dir)), specular_exp);
				}

				out += specular_factor * glm::vec4(specular_colour, 0.0f);
			}

			// Do reflection ray
			// Check if light hits this position
			//if (iteration == 0 && glm::dot(view_dir, normal) < 0.0f)
			//{
			//	const float outset_size = 0.02f;

			//	glm::vec3 reflection = glm::reflect(view_dir, normal);

			//	ray light_ray;
			//	light_ray.direction = glm::normalize(reflection);

			//	// The origin is outset a little so we don't just hit the same voxel
			//	light_ray.origin = hit_pos + light_ray.direction * outset_size;

			//	float hit_t;						// Resultant hit t value from raycast
			//	glm::vec3 hit_pos;					// Resultant hit position from raycast
			//	const int32_t* hit_parent;			// Resultant hit parent voxel from raycast
			//	int32_t hit_idx;					// Resultant hit child index from raycast
			//	int32_t hit_scale;					// Resultant hit scale from raycast

			//	// Do raycast
			//	raycast(root, &light_ray, &hit_t, &hit_pos, &hit_parent, &hit_idx, &hit_scale);

			//	// If we hit a voxel in the tree
			//	if (hit_scale < MAX_SCALE)
			//	{
			//		//printf("reflection\n");
			//		//out += shade(rendering_attributes, hit_t, hit_pos, root, hit_parent, hit_idx, hit_scale, 100);
			//		//printf("done reflection\n");
			//		out += glm::vec4(0.3f, 0.3f, 0.3f, 0);
			//	}
			//}
			//{
			//	// Calculate reflection ray
			//	glm::vec3 reflection = glm::reflect(view_dir, normal);

			//	const float outset_size = 0.02f;

			//	ray reflection_ray;
			//	reflection_ray.direction = glm::normalize(reflection);

			//	// The origin is outset a little so we don't just hit the same voxel
			//	reflection_ray.origin = hit_pos + reflection_ray.direction * outset_size;

			//	float hit_t;							// Resultant hit t value from raycast
			//	glm::vec3 hit_pos;						// Resultant hit position from raycast
			//	const int32_t* hit_parent;				// Resultant hit parent voxel from raycast
			//	int32_t hit_idx;						// Resultant hit child index from raycast
			//	int32_t hit_scale;						// Resultant hit scale from raycast

			//	// Do raycast
			//	raycast(root + child_desc_size_ints, &reflection_ray, &hit_t, &hit_pos, &hit_parent, &hit_idx, &hit_scale);

			//	// If we hit a voxel in the tree
			//	if (hit_scale < MAX_SCALE)
			//	{
			//		out += shade(rendering_attributes, hit_t, hit_pos, root, hit_parent, hit_idx, hit_scale, 100);
			//	}
			//}

			// Clamp to between 0 and 1
			out = glm::clamp(out, 0.0f, 1.0f);
			
			return out;
		}

		__device__ const raw_attachment* lookupRawAttachment(const int32_t* root,
													const int32_t* parent, int child_idx)
		{
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

			const raw_attachment* attachment = (raw_attachment*)((uintptr_t)root + *raw_lookup);

			return attachment;
		}
	}
}