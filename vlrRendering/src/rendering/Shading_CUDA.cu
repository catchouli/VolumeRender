#include "rendering/Shading.h"

#include "maths/Normal.h"
#include "rendering/child_desc.h"
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

		__device__ int shade(const rendering_attributes_t rendering_attributes, glm::vec3 viewdir,
			float hit_t, glm::vec3 hit_pos,
			const int* hit_parent, int hit_idx, int hit_scale)
		{
			// Calculate view direction
			glm::vec3 view_dir = glm::normalize(rendering_attributes.origin - hit_pos);

			// Parent child desc
			child_desc_word_1 parent_c_desc = *(child_desc_word_1*)&hit_parent[0];

			// Load normal
			glm::vec3 normal = glm::normalize(decompressNormal(hit_parent[1]));

			// Load colour
			col com_col = *(col*)&hit_parent[2];

			// Decompress colour
			glm::vec3 colour(com_col.r, com_col.g, com_col.b);
			colour = (colour) / 255.0f;
			colour = glm::vec3(1.0f, 1.0f, 1.0f);

			// Calculate lighting
			glm::vec3 out(0, 0, 0);

			// Calculate ambient
			out += rendering_attributes.ambient_colour * colour;

			// For each light
			for (int i = 0; i < rendering_attributes.light_count; ++i)
			{
				const light_t& light = rendering_attributes.lights[i];

				// Attenuation (1.0f for directional lights)
				float attenuation = 1.0f;

				// Light direction (light.direction for directional lights)
				glm::vec3 light_dir = glm::normalize(light.direction);

				// Calculate direction and attentuation for non directional lights
				if (light.type != LightTypes::DIRECTIONAL)
				{
					// Calculate light direction
					glm::vec3 light_diff = hit_pos - light.position;
					light_dir = glm::normalize(light_diff);

					// Calculate light distance
					float distance = glm::length(light_diff);

					// Calculate attenuation
					attenuation = 1.0f /
						(light.constant_att + light.linear_att * distance +
						light.quadratic_att * distance * distance);

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

				// Calculate diffuse reflection
				float diffuse_factor = attenuation * fmaxf(0.0f, glm::dot(normal, -light_dir));
				glm::vec3 diffuse_colour = colour * light.diffuse;

				out += diffuse_factor * diffuse_colour;

				// Calculate specular reflection
				float specular_factor = 0.0f;
				glm::vec3 specular_colour = colour * light.specular;

				float specular_exp = 32.0f;

				// If the normal faces the light
				if (glm::dot(normal, light_dir) < 0.0f)
				{
					glm::vec3 reflection = glm::reflect(light_dir, normal);
					specular_factor = powf(fmaxf(0.0f, glm::dot(reflection, view_dir)), specular_exp);
				}

				out += specular_factor * specular_colour;
			}

			// Clamp to between 0 and 1
			out = glm::clamp(out, 0.0f, 1.0f);

			// Convert out to colour
			col out_col = { 0 };
			out_col.r = (unsigned char)(out.x * 255.0f);
			out_col.g = (unsigned char)(out.y * 255.0f);
			out_col.b = (unsigned char)(out.z * 255.0f);

			// Write to buffer
			return *(unsigned int*)&out_col;
		}
	}
}