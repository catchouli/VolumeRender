#include "rendering/Raycast.h"

#ifndef VLR_RAYCAST_CPU

// Define cudacc to get rid of some intellisense errors
#ifndef __CUDACC__
#	define __CUDACC__
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "maths/Matrix.h"
#include "maths/Normal.h"
#include "maths/Types.h"
#include "resources/Octree.h"
#include "rendering/Rendering.h"
#include "rendering/Raycast.h"
#include "rendering/Shading.h"
#include "util/Util.h"
#include "util/CUDAUtil.h"

#define float_as_int __float_as_int
#define int_as_float __int_as_float

namespace vlr
{
	namespace rendering
	{
		__device__ void raycast(const int32_t* tree, const rendering::ray* ray,
									StackEntry* stack, RaycastHit* raycastHit)
		{
			// The smallest possible positive nonzero floating point number
			const float min_float = exp2f(-MAX_SCALE);

			// Get ray position and direction
			const glm::vec3& origin = ray->origin;
			glm::vec3 dir = ray->direction;

			// Eliminate small (zero) direction values to avoid division by zero
			if (fabsf(dir.x) < min_float)
				dir.x = copysignf(min_float, dir.x);
			if (fabsf(dir.y) < min_float)
				dir.y = copysignf(min_float, dir.y);
			if (fabsf(dir.z) < min_float)
				dir.z = copysignf(min_float, dir.z);

			// Precompute ray equation coefficient and constant for
			// P(t) = o + td
			// Rearranged using inverse expressions of x, y, and z
			float tx_coef = 1.0f / -fabs(dir.x);
			float ty_coef = 1.0f / -fabs(dir.y);
			float tz_coef = 1.0f / -fabs(dir.z);

			float tx_constant = tx_coef * origin.x;
			float ty_constant = ty_coef * origin.y;
			float tz_constant = tz_coef * origin.z;

			// Mirror coordinate system
			// Makes algorithm have fewer cases since ray direction is no longer important
			// + allows for some optimisations later on
			int32_t dir_mask = 7;

			if (dir.x > 0.0f)
			{
				dir_mask ^= (1 << 0);
				tx_constant = 3.0f * tx_coef - tx_constant;
			}

			if (dir.y > 0.0f)
			{
				dir_mask ^= (1 << 1);
				ty_constant = 3.0f * ty_coef - ty_constant;
			}

			if (dir.z > 0.0f)
			{
				dir_mask ^= (1 << 2);
				tz_constant = 3.0f * tz_coef - tz_constant;
			}

			// Compute span of t values for root
			float tx_min = 2.0f * tx_coef - tx_constant;
			float ty_min = 2.0f * ty_coef - ty_constant;
			float tz_min = 2.0f * tz_coef - tz_constant;

			float tx_max = tx_coef - tx_constant;
			float ty_max = ty_coef - ty_constant;
			float tz_max = tz_coef - tz_constant;

			float t_min = fmaxf(fmaxf(tx_min, ty_min), tz_min);
			float t_max = fminf(fminf(tx_max, ty_max), tz_max);

			t_min = fmaxf(t_min, 0.0f);

			float h = t_max;

			// Get root node
			const int32_t* parent = tree + child_desc_size_ints;

			// Evaluate root at centre to get first child node
			int32_t idx = 0;
			float3 pos = { 1.0f, 1.0f, 1.0f };

			int32_t scale = MAX_SCALE - 1;
			float scale_exp2 = 0.5f;

			float tx_centre = 1.5f * tx_coef - tx_constant;
			float ty_centre = 1.5f * ty_coef - ty_constant;
			float tz_centre = 1.5f * tz_coef - tz_constant;

			// Compare centre values against t_min to obtain
			// each bit of idx
			if (tx_centre > t_min)
			{
				idx ^= (1 << 0);
				pos.x = 1.5f;
			}

			if (ty_centre > t_min)
			{
				idx ^= (1 << 1);
				pos.y = 1.5f;
			}

			if (tz_centre > t_min)
			{
				idx ^= (1 << 2);
				pos.z = 1.5f;
			}

			// Cached child descriptor
			int2 child_descriptor = make_int2(0, 0);
			
			// Run until we pop the root voxel
			while (scale < MAX_SCALE)
			{
				// Fetch child descriptor if not valid
				if (child_descriptor.x == 0)
					child_descriptor = *(int2*)parent;

				// Calculate t_max for child
				tx_max = pos.x * tx_coef - tx_constant;
				ty_max = pos.y * ty_coef - ty_constant;
				tz_max = pos.z * tz_coef - tz_constant;

				float t_c_max = fminf(fminf(tx_max, ty_max), tz_max);

				// Mirror idx to get child index
				int32_t child_idx = idx ^ dir_mask;
				int32_t child_mask = child_descriptor.x << child_idx;

				//const raw_attachment* attachment = lookupRawAttachment(tree, parent, 0);
				//glm::vec3 normal = glm::normalize(decompressNormal(attachment->normal));

				// Process voxel if existent and the current span of t values is valid
				if ((child_mask & 0x8000) != 0 && t_min <= t_max)
				// Check the dot product of the normal and the ray direction (basically front face culling)
				//if (glm::dot(glm::vec3(0, 0, -1), normal) > 0.0f)
				//if (attachment != nullptr)
				{
					// TODO:
					// Check if voxel is small enough to terminate traversal
					// (Efficient sparse voxel octrees, Karras and Laine)

					// Find the intersection of t_max and t_c_max
					float tvmax = fminf(t_max, t_c_max);

					// Evaluate child at centre
					float tx_centre = 0.5f * scale_exp2 * tx_coef + tx_max;
					float ty_centre = 0.5f * scale_exp2 * ty_coef + ty_max;
					float tz_centre = 0.5f * scale_exp2 * tz_coef + tz_max;

					// TODO:
					// Implement contours
					// (Efficient sparse voxel octrees, Karras and Laine)

					// Descend if the resulting span is non-zero
					if (t_min <= tvmax)
					{
						// Terminate if this is a leaf voxel
						if ((child_mask & 0x80) == 0)
						{
							break;
						}

						// Write parent voxel and t_max to stack
						if (t_c_max < h)
						{
							stack[scale].parent = parent;
							stack[scale].t_max = t_max;
						}

						// Store h value to eliminate unnecessary stack writes
						h = t_c_max;

						// Update parent voxel
						int32_t ofs = (uint32_t)(child_descriptor.x) >> 17;

						// If this is a far pointer, load it
						if ((child_descriptor.x & 0x10000) != 0)
							ofs = parent[ofs * child_desc_size_ints];

						ofs += get_child_index(child_mask & 0x7F);
						parent += child_desc_size_ints * ofs;

						// Update scale
						scale--;
						scale_exp2 *= 0.5f;

						// Get first child
						idx = 0;

						// Compare t value at centre to get new idx
						if (tx_centre > t_min)
						{
							idx ^= 1;
							pos.x += scale_exp2;
						}

						if (ty_centre > t_min)
						{
							idx ^= 2;
							pos.y += scale_exp2;
						}

						if (tz_centre > t_min)
						{
							idx ^= 4;
							pos.z += scale_exp2;
						}

						// Update max t value
						t_max = tvmax;

						// Invalidate cache child descriptor
						child_descriptor.x = 0;

						continue;
					}
				}

				// Advance the ray
				int32_t step_mask = 0;

				if (tx_max <= t_c_max)
				{
					step_mask ^= (1 << 0);
					pos.x -= scale_exp2;
				}

				if (ty_max <= t_c_max)
				{
					step_mask ^= (1 << 1);
					pos.y -= scale_exp2;
				}
				if (tz_max <= t_c_max)
				{
					step_mask ^= (1 << 2);
					pos.z -= scale_exp2;
				}

				// Update t_min
				t_min = t_c_max;

				// Flip idx
				idx ^= step_mask;

				// Check that direction of flips agree with ray direction
				if ((idx & step_mask) != 0)
				{
					// Pop
					// Find the highest differing bit between pos and oldPos
					uint32_t differing_bits = 0;

					// Opaque bitwise wizardry courtesy of Efficient Sparse Voxel Octrees (Laine and Karras)
					// Get differing bits between each component of pos and oldpos (oldpos.x ^ oldpos.y etc)
					// Then or together the ones which have changed to obtain which bits differ between all three
					if ((step_mask & (1 << 0)) != 0) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.x + scale_exp2);
					if ((step_mask & (1 << 1)) != 0) differing_bits |= __float_as_int(pos.y) ^ __float_as_int(pos.y + scale_exp2);
					if ((step_mask & (1 << 2)) != 0) differing_bits |= __float_as_int(pos.z) ^ __float_as_int(pos.z + scale_exp2);

					// Calculate the scale (the position of the greatest bit)
					scale = (__float_as_int((float)differing_bits) >> 23) - 127;

					// Calculate scale_exp2 (2^(scale - maxScale))
					scale_exp2 = __int_as_float((scale - MAX_SCALE + 127) << 23);

					// Restore parent voxel from the stack.
					StackEntry stackEntry = stack[scale];
					parent = stackEntry.parent;
					t_max = stackEntry.t_max;

					// Get rid of pos values under new scale
					int32_t temp_x = __float_as_int(pos.x) >> scale;
					int32_t temp_y = __float_as_int(pos.y) >> scale;
					int32_t temp_z = __float_as_int(pos.z) >> scale;

					pos.x = __int_as_float(temp_x << scale);
					pos.y = __int_as_float(temp_y << scale);
					pos.z = __int_as_float(temp_z << scale);

					idx = (temp_x & 1) | ((temp_y & 1) << 1) | ((temp_z & 1) << 2);

					// Prevent unnecessary stack writes
					h = 0.0f;
				}
			}

			// Undo mirroring of the coordinate system
			if ((dir_mask & (1 << 0)) == 0) pos.x = 3.0f - scale_exp2 - pos.x;
			if ((dir_mask & (1 << 1)) == 0) pos.y = 3.0f - scale_exp2 - pos.y;
			if ((dir_mask & (1 << 2)) == 0) pos.z = 3.0f - scale_exp2 - pos.z;

			// Output return values
			// Output t of hit
			raycastHit->hit_t = t_min;

			// Output position of hit
			raycastHit->hit_pos.x = fminf(fmaxf(origin.x + t_min * dir.x, pos.x + min_float), pos.x + scale_exp2 - min_float);
			raycastHit->hit_pos.y = fminf(fmaxf(origin.y + t_min * dir.y, pos.y + min_float), pos.y + scale_exp2 - min_float);
			raycastHit->hit_pos.z = fminf(fmaxf(origin.z + t_min * dir.z, pos.z + min_float), pos.z + scale_exp2 - min_float);

			// Output parent of hit voxel
			raycastHit->hit_parent = parent;

			// Output child index of hit voxel
			raycastHit->hit_idx = idx ^ (dir_mask ^ 7);
			raycastHit->hit_pos_internal = *(glm::vec3*)&pos;

			// Output scale of hit voxel
			raycastHit->hit_scale = scale;
		}

		// TODO: restore old state
		__device__ void raycast_empty(const int32_t* tree, const rendering::ray* ray, StackEntry* stack,
								RaycastHit* raycastHit, const RaycastHit* old_hit)
		{
			const float MAX_REFRACTION_STEP = 100.0f;

			// The smallest possible positive nonzero floating point number
			const float min_float = exp2f(-MAX_SCALE);

			// Get ray position and direction
			const glm::vec3& origin = ray->origin;
			glm::vec3 dir = ray->direction;

			// Eliminate small (zero) direction values to avoid division by zero
			if (fabsf(dir.x) < min_float)
				dir.x = copysignf(min_float, dir.x);
			if (fabsf(dir.y) < min_float)
				dir.y = copysignf(min_float, dir.y);
			if (fabsf(dir.z) < min_float)
				dir.z = copysignf(min_float, dir.z);

			// Precompute ray equation coefficient and constant for
			// P(t) = o + td
			// Rearranged using inverse expressions of x, y, and z
			float tx_coef = 1.0f / -fabs(dir.x);
			float ty_coef = 1.0f / -fabs(dir.y);
			float tz_coef = 1.0f / -fabs(dir.z);

			float tx_constant = tx_coef * origin.x;
			float ty_constant = ty_coef * origin.y;
			float tz_constant = tz_coef * origin.z;

			// Mirror coordinate system
			// Makes algorithm have fewer cases since ray direction is no longer important
			// + allows for some optimisations later on
			int32_t dir_mask = 7;

			if (dir.x > 0.0f)
			{
				dir_mask ^= (1 << 0);
				tx_constant = 3.0f * tx_coef - tx_constant;
			}

			if (dir.y > 0.0f)
			{
				dir_mask ^= (1 << 1);
				ty_constant = 3.0f * ty_coef - ty_constant;
			}

			if (dir.z > 0.0f)
			{
				dir_mask ^= (1 << 2);
				tz_constant = 3.0f * tz_coef - tz_constant;
			}

			// Compute span of t values for root
			float tx_min = 2.0f * tx_coef - tx_constant;
			float ty_min = 2.0f * ty_coef - ty_constant;
			float tz_min = 2.0f * tz_coef - tz_constant;

			float tx_max = tx_coef - tx_constant;
			float ty_max = ty_coef - ty_constant;
			float tz_max = tz_coef - tz_constant;

			float t_min = fmaxf(fmaxf(tx_min, ty_min), tz_min);
			float t_max = fminf(fminf(tx_max, ty_max), tz_max);

			t_min = fmaxf(t_min, 0.0f);
			t_max = fminf(t_max, t_min + MAX_REFRACTION_STEP);

			float h = t_max;

			// Get root node
			const int32_t* parent = tree + child_desc_size_ints;

			// Evaluate root at centre to get first child node
			int32_t idx = 0;
			float3 pos = { 1.0f, 1.0f, 1.0f };

			// Store old parent & idx in case we need to return them
			const int32_t* old_parent = parent;
			int32_t old_idx = idx;
			float3 old_pos = pos;

			int32_t scale = MAX_SCALE - 1;
			float scale_exp2 = 0.5f;

			float tx_centre = 1.5f * tx_coef - tx_constant;
			float ty_centre = 1.5f * ty_coef - ty_constant;
			float tz_centre = 1.5f * tz_coef - tz_constant;

			// Compare centre values against t_min to obtain
			// each bit of idx
			if (tx_centre > t_min)
			{
				idx ^= (1 << 0);
				pos.x = 1.5f;
			}

			if (ty_centre > t_min)
			{
				idx ^= (1 << 1);
				pos.y = 1.5f;
			}

			if (tz_centre > t_min)
			{
				idx ^= (1 << 2);
				pos.z = 1.5f;
			}

			// Cached child descriptor
			int2 child_descriptor = make_int2(0, 0);
			
			// Run until we pop the root voxel
			while (scale < MAX_SCALE)
			{
				// Fetch child descriptor if not valid
				if (child_descriptor.x == 0)
					child_descriptor = *(int2*)parent;

				// Calculate t_max for child
				tx_max = pos.x * tx_coef - tx_constant;
				ty_max = pos.y * ty_coef - ty_constant;
				tz_max = pos.z * tz_coef - tz_constant;

				float t_c_max = fminf(fminf(tx_max, ty_max), tz_max);

				// Mirror idx to get child index
				int32_t child_idx = idx ^ dir_mask;
				int32_t child_mask = child_descriptor.x << child_idx;
				//const raw_attachment* attachment = lookupRawAttachment(tree, parent, 0);
				//glm::vec3 normal = glm::normalize(decompressNormal(attachment->normal));

				if ((child_mask & 0x8000) != 0 && t_min <= t_max)
				{
					// Store old parent & idx in case we need to return them
					old_parent = parent;
					old_idx = idx;
					old_pos = pos;

					// TODO: contours (laine and karras)
					// If this voxel has a contour
					int contour_mask = child_descriptor.y;
					if ((contour_mask & 0x80) != 0)
					{

					}

					// Find the intersection of t_max and t_c_max
					float tvmax = fminf(t_max, t_c_max);

					// Evaluate child at centre
					float tx_centre = 0.5f * scale_exp2 * tx_coef + tx_max;
					float ty_centre = 0.5f * scale_exp2 * ty_coef + ty_max;
					float tz_centre = 0.5f * scale_exp2 * tz_coef + tz_max;

					// TODO:
					// Implement contours
					// (Efficient sparse voxel octrees, Karras and Laine)

					// Descend if the resulting span is non-zero
					if (t_min <= tvmax)
					{
						// Push if this is not a leaf voxel
						if ((child_mask & 0x80) != 0)
						{
							// Write parent voxel and t_max to stack
							if (t_c_max < h)
							{
								stack[scale].parent = parent;
								stack[scale].t_max = t_max;
							}

							// Store h value to eliminate unnecessary stack writes
							h = t_c_max;

							// Update parent voxel
							int32_t ofs = (uint32_t)(child_descriptor.x) >> 17;

							// If this is a far pointer, load it
							if ((child_descriptor.x & 0x10000) != 0)
								ofs = parent[ofs * child_desc_size_ints];

							ofs += get_child_index(child_mask & 0x7F);
							parent += child_desc_size_ints * ofs;

							// Update scale
							scale--;
							scale_exp2 *= 0.5f;

							// Get first child
							idx = 0;

							// Compare t value at centre to get new idx
							if (tx_centre > t_min)
							{
								idx ^= 1;
								pos.x += scale_exp2;
							}

							if (ty_centre > t_min)
							{
								idx ^= 2;
								pos.y += scale_exp2;
							}

							if (tz_centre > t_min)
							{
								idx ^= 4;
								pos.z += scale_exp2;
							}

							// Update max t value
							t_max = tvmax;

							// Invalidate cache child descriptor
							child_descriptor.x = 0;

							continue;
						}
					}
				}

				// Advance the ray
				int32_t step_mask = 0;

				if (tx_max <= t_c_max)
				{
					step_mask ^= (1 << 0);
					pos.x -= scale_exp2;
				}

				if (ty_max <= t_c_max)
				{
					step_mask ^= (1 << 1);
					pos.y -= scale_exp2;
				}
				if (tz_max <= t_c_max)
				{
					step_mask ^= (1 << 2);
					pos.z -= scale_exp2;
				}

				// Update t_min
				t_min = t_c_max;

				// Flip idx
				idx ^= step_mask;

				//printf("%f\n", t_c_max);

				// Check that direction of flips agree with ray direction
				if ((idx & step_mask) != 0)
				{
					// Pop
					// Find the highest differing bit between pos and oldPos
					uint32_t differing_bits = 0;

					// Opaque bitwise wizardry courtesy of Efficient Sparse Voxel Octrees (Laine and Karras)
					// Get differing bits between each component of pos and oldpos (oldpos.x ^ oldpos.y etc)
					// Then or together the ones which have changed to obtain which bits differ between all three
					if ((step_mask & (1 << 0)) != 0) differing_bits |= float_as_int(pos.x) ^ float_as_int(pos.x + scale_exp2);
					if ((step_mask & (1 << 1)) != 0) differing_bits |= float_as_int(pos.y) ^ float_as_int(pos.y + scale_exp2);
					if ((step_mask & (1 << 2)) != 0) differing_bits |= float_as_int(pos.z) ^ float_as_int(pos.z + scale_exp2);

					// Calculate the scale (the position of the greatest bit)
					scale = (float_as_int((float)differing_bits) >> 23) - 127;

					// Calculate scale_exp2 (2^(scale - maxScale))
					scale_exp2 = int_as_float((scale - MAX_SCALE + 127) << 23);

					// Restore parent voxel from the stack.
					StackEntry stackEntry = stack[scale];
					parent = stackEntry.parent;
					t_max = stackEntry.t_max;

					// Get rid of pos values under new scale
					int32_t temp_x = float_as_int(pos.x) >> scale;
					int32_t temp_y = float_as_int(pos.y) >> scale;
					int32_t temp_z = float_as_int(pos.z) >> scale;

					pos.x = int_as_float(temp_x << scale);
					pos.y = int_as_float(temp_y << scale);
					pos.z = int_as_float(temp_z << scale);

					// Compute new idx
					idx = (temp_x & 1) | ((temp_y & 1) << 1) | ((temp_z & 1) << 2);

					// Invalidate cached descriptor
					child_descriptor.x = 0;

					// Prevent unnecessary stack writes
					h = 0.0f;
				}

				if (scale != 23)
				{
					// Read child desc
					int32_t cdesc = *(int32_t*)parent;

					// If this voxel does not exist
					if (((cdesc << (idx ^ dir_mask)) & 0x8000) == 0)
					{
						// Restore last existent voxel and return
						idx = old_idx;
						pos = old_pos;
						parent = old_parent;

						break;
					}
				}
			}

			// Undo mirroring of the coordinate system
			if ((dir_mask & (1 << 0)) == 0) pos.x = 3.0f - scale_exp2 - pos.x;
			if ((dir_mask & (1 << 1)) == 0) pos.y = 3.0f - scale_exp2 - pos.y;
			if ((dir_mask & (1 << 2)) == 0) pos.z = 3.0f - scale_exp2 - pos.z;

			// Output return values
			// Output t of hit
			raycastHit->hit_t = t_min;

			// Output position of hit
			raycastHit->hit_pos.x = fminf(fmaxf(origin.x + t_min * dir.x, pos.x + min_float), pos.x + scale_exp2 - min_float);
			raycastHit->hit_pos.y = fminf(fmaxf(origin.y + t_min * dir.y, pos.y + min_float), pos.y + scale_exp2 - min_float);
			raycastHit->hit_pos.z = fminf(fmaxf(origin.z + t_min * dir.z, pos.z + min_float), pos.z + scale_exp2 - min_float);

			// Output parent of hit voxel
			raycastHit->hit_parent = parent;

			// Output child index of hit voxel
			raycastHit->hit_idx = idx ^ (dir_mask ^ 7);
			raycastHit->hit_pos_internal = *(glm::vec3*)&pos;

			// Output scale of hit voxel
			raycastHit->hit_scale = scale;
		}
	}
}

#endif /* VLR_RAYCAST_CPU */
