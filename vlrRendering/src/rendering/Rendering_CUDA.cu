#include "rendering/Rendering.h"

#include "maths/Colour.h"
#include "rendering/Shading.h"
#include "rendering/rendering_attributes.h"
#include "resources/Image.h"
#include "util/Util.h"
#include "util/CUDAUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include <stdint.h>

namespace vlr
{
	namespace rendering
	{
		template <int recursions>
		__device__ inline glm::vec4 shade(const rendering_attributes_t rendering_attributes, const ray& eye_ray, RaycastHit& hit,
			StackEntry* stack, const int32_t* tree, float index)
		{
			const float min_float = exp2f(-MAX_SCALE);

			raw_attachment_uncompressed shading_attributes;

			// Look up raw attachment
			const raw_attachment* hit_voxel_attachment =
				lookupRawAttachment(tree, hit.hit_parent, hit.hit_idx);
				
			// Unpack raw attachment
			unpack_raw_attachment(*hit_voxel_attachment, shading_attributes);

			// Normalise normal
			shading_attributes.normal = glm::normalize(shading_attributes.normal);

			// Result
			glm::vec4 out(0, 0, 0, 0);

			// Calculate view direction
			glm::vec3 view_dir = eye_ray.direction;

			// Calculate lighting
			glm::vec3 colour = glm::vec3(shading_attributes.colour);
			glm::vec3 normal = shading_attributes.normal;
			
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
					glm::vec3 light_diff = hit.hit_pos - light.position;
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
				if (rendering_attributes.settings.enable_shadows)
				{
					if (glm::dot(light_dir, normal) < 0)
					{
						ray light_ray;
						light_ray.direction = -light_dir;

						// The origin is outset a little so we don't just hit the same voxel
						light_ray.origin = hit.hit_pos +
							light_ray.direction * (float)HALF_SQRT_2 * exp2f(hit.hit_scale - MAX_SCALE);

						RaycastHit shadowRayHit;

						// Do raycast
						raycast(tree, &light_ray, stack, &shadowRayHit);

						// If we hit a voxel in the tree
						if (shadowRayHit.hit_scale < MAX_SCALE && shadowRayHit.hit_t < light_distance)
						{
							continue;
						}
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

			// TODO: use fresnel approximation to combine reflection and refraction
			// Handle refraction
			if (rendering_attributes.settings.enable_refraction
				&& shading_attributes.colour.a < 1.0f)
			{
				float one_minus_alpha = 1.0f - shading_attributes.colour.a;

				out *= shading_attributes.colour.a;

				// Get surface normal
				glm::vec3 normal = shading_attributes.normal;

				// Reverse normal if it doesn't face the ray
				if (glm::dot(normal, eye_ray.direction) > 0.0f)
				{
					normal = -normal;
				}

				// Calculate refraction ray
				ray refraction_ray;
				refraction_ray.direction = glm::refract(eye_ray.direction, normal, index / shading_attributes.refractive_index);

				// The origin is inset a little so we're sure to hit the right voxel
				refraction_ray.origin = hit.hit_pos +
					refraction_ray.direction * (float)HALF_SQRT_2 * exp2f(hit.hit_scale - MAX_SCALE);

				// Cast ray through solid for refraction
				RaycastHit refraction_hit;

				if (rendering_attributes.settings.refraction_mode == RefractionModes::CONTINUOUS)
				{
					// Continuous refraction
					raycast_empty(tree, &refraction_ray, stack, &refraction_hit, nullptr);
				}
				else
				{
					// Discrete refraction
					// Cast rays backwards at intervals
					float refractive_index = shading_attributes.refractive_index;

					ray discrete_refraction_ray;
					discrete_refraction_ray.origin = refraction_ray.origin +
						refraction_ray.direction * (float)HALF_SQRT_2 * exp2f(hit.hit_scale - MAX_SCALE);
					discrete_refraction_ray.direction = -refraction_ray.direction;

					int refracted_times = 0;
					for (int i = 0;
						i < rendering_attributes.settings.refraction_discrete_steps_max;
						++i)
					{
						// Raycast back along ray
						raycast(tree, &discrete_refraction_ray, stack, &refraction_hit);

						// No hit, hit the edge of the tree
						if (refraction_hit.hit_scale == MAX_SCALE)
							break;

						// If hit_t == 0, we haven't found a surface (the ray started inside a volume)
						if (refraction_hit.hit_t > min_float)
							break;

						// Look up raw attachment
						const raw_attachment* hit_voxel_refraction =
							lookupRawAttachment(tree, refraction_hit.hit_parent, refraction_hit.hit_idx);
				
						// Unpack raw attachment
						raw_attachment_uncompressed refraction_attachment;
						unpack_raw_attachment(*hit_voxel_refraction, refraction_attachment);

						// Check refractive index
						if (refraction_attachment.refractive_index != refractive_index)
						{
							refracted_times++;

							float eta = refractive_index / refraction_attachment.refractive_index;

							glm::vec3 normal = glm::normalize(refraction_attachment.normal);

							if (glm::dot(normal, refraction_ray.direction) > 0.0f)
								normal = -normal;

							refraction_ray.direction = glm::refract(refraction_ray.direction, normal,
								eta);

							discrete_refraction_ray.direction = -refraction_ray.direction;

							refractive_index = refraction_attachment.refractive_index;
						}

						// Increment pos by step
						discrete_refraction_ray.origin += refraction_ray.direction *
							rendering_attributes.settings.refraction_discrete_step;
					}
				}
				
				if (refraction_hit.hit_scale >= MAX_SCALE)
				{
					out += one_minus_alpha * glm::vec4(rendering_attributes.ambient_colour, 0.0f);
				}
				else
				{
					// Look up raw attachment
					const raw_attachment* hit_voxel_refraction =
						lookupRawAttachment(tree, refraction_hit.hit_parent, refraction_hit.hit_idx);
				
					// Unpack raw attachment
					raw_attachment_uncompressed refraction_attachment;
					unpack_raw_attachment(*hit_voxel_refraction, refraction_attachment);

					glm::vec3 normal = glm::normalize(refraction_attachment.normal);

					if (glm::dot(refraction_ray.direction, normal) > 0.0f)
						normal = -normal;

					// Cast ray to find next surface after refraction
					refraction_ray.direction =
						glm::refract(refraction_ray.direction,
									normal,
									refraction_attachment.refractive_index / index);

					refraction_ray.origin = refraction_hit.hit_pos +
						refraction_ray.direction * (float)HALF_SQRT_2 * exp2f(hit.hit_scale - MAX_SCALE);
					
					raycast(tree, &refraction_ray, stack, &refraction_hit);

					if (refraction_hit.hit_scale < MAX_SCALE)
					{
						out += one_minus_alpha *
							shade<recursions+1>(rendering_attributes, refraction_ray, refraction_hit,
												stack, tree);
					}
					else
					{
						out += one_minus_alpha * glm::vec4(rendering_attributes.ambient_colour, 0.0f);
					}
				}
			}

			// Handle reflection
			if (rendering_attributes.settings.enable_reflection
				&& shading_attributes.reflectivity > 0.0f)
			{
				float one_minus_reflectivity = 1.0f - shading_attributes.reflectivity;

				out *= one_minus_reflectivity;

				// Calculate reflection ray
				ray reflection_ray;
				reflection_ray.direction = glm::reflect(eye_ray.direction, shading_attributes.normal);

				reflection_ray.origin = hit.hit_pos +
					reflection_ray.direction * (float)HALF_SQRT_2 * exp2f(hit.hit_scale - MAX_SCALE);

				// Cast reflection ray
				RaycastHit reflection_hit;

				raycast(tree, &reflection_ray, stack, &reflection_hit);

				if (reflection_hit.hit_scale < MAX_SCALE)
				{
					out += shading_attributes.reflectivity *
						shade<recursions+1>(rendering_attributes, reflection_ray, reflection_hit,
											stack, tree);
				}
				else
				{
					out += shading_attributes.reflectivity * rendering_attributes.clear_colour;
				}
			}

			// Clamp to between 0 and 1
			out = glm::clamp(out, 0.0f, 1.0f);
			
			return out;
		}

		__device__ __host__ ray screenPointToRay(int32_t x, int32_t y,
			const rendering_attributes_t& rendering_attributes)
		{
			ray ret;

			// Origin is the camera position
			ret.origin = rendering_attributes.origin;

			// Calculate x position on viewport
			float width = (float)rendering_attributes.viewport.w;
			float height = (float)rendering_attributes.viewport.h;
			float oneOverWidth = 1.0f / width;
			float oneOverHeight = 1.0f / height;

			float normx = x * oneOverWidth;
			float normy = y * oneOverHeight;
			
			// Multiply viewport position by mvp to get world position
			glm::vec4 viewportPos;
			viewportPos.x = normx * 2.0f - 1.0f;
			viewportPos.y = normy * 2.0f - 1.0f;
			viewportPos.z = 1.0f;
			viewportPos.w = 1.0f;

			ret.direction = glm::normalize(glm::vec3(viewportPos * rendering_attributes.mvp));

			return ret;
		}

		__global__ void cudaRenderOctree(const int32_t* tree, int32_t* pixel_buffer,
			GLfloat* depth_buffer, const rendering_attributes_t rendering_attributes)
		{
			const viewport& viewport = rendering_attributes.viewport;
			
			const int32_t width = viewport.w;		// Viewport width
			const int32_t height = viewport.h;		// Viewport height
			
			ray eye_ray;							// Eye ray for this kernel
			int32_t x, y;							// x and y of this kernel's pixel
			int32_t* pixel;							// Pointer to this kernel's pixel
			GLfloat* depth_pixel;					// Pointer to this kernel's pixel in the depth buffer
			
			StackEntry stack[MAX_SCALE + 1];		// Stack for parent voxels
			RaycastHit raycastHit;					// Raycast result

			// Calculate x and y
			// (each block works on a blockDim.x * blockDim.y square of pixels, so
			// p = block indices * block dimensions + thread indices;
			x = blockIdx.x * blockDim.x + threadIdx.x;
			y = blockIdx.y * blockDim.y + threadIdx.y;

			// Due to the const block dimensions, some pixels outside of the screen
			// may have kernels spawned, but no < 0 values are possible
			if (x >= width || y >= height)
				return;

			// Get the pixel in the pixel buffer using the x and y coordinate
			int offset = width * y + x;
			pixel = pixel_buffer + offset;
			depth_pixel = depth_buffer + offset;

			// Calculate eye ray for pixel
			eye_ray = screenPointToRay(x, y, rendering_attributes);

			// Clear to black
			*pixel = compressColour(rendering_attributes.clear_colour);

			// Write far value into depth buffer
			if (depth_buffer != nullptr)
				*depth_pixel = 1.0f;

			// Do raycast
			raycast(tree, &eye_ray, stack, &raycastHit);

			// If we hit a voxel in the tree
			if (raycastHit.hit_scale < MAX_SCALE)
			{
				// Calculate depth buffer value
				glm::vec4 ndc_pos = rendering_attributes.mvp * glm::vec4(raycastHit.hit_pos, 1.0f);
				float depth = ((ndc_pos.z / ndc_pos.w) + 1.0f) * 0.5f;
				
				// Clamp to -1.0 .. 1.0 and write to depth buffer
				if (depth_buffer != nullptr)
					*depth_pixel = depth;

				// Look up raw attachment
				const raw_attachment* hit_voxel_attachment =
					lookupRawAttachment(tree, raycastHit.hit_parent, raycastHit.hit_idx);
				
				// Decompress raw attachment
				raw_attachment_uncompressed shading_attributes;
				unpack_raw_attachment(*hit_voxel_attachment, shading_attributes);

				// Shade fragment recursively
				glm::vec4 out_colour = shade(rendering_attributes, eye_ray, raycastHit, stack,
												tree);

				// Write to colour buffer
				*pixel = compressColour(out_colour);
			}
		}

		void renderOctree(const int32_t* treeGpu, rendering_attributes_t rendering_attributes)
		{
			const viewport& viewport = rendering_attributes.viewport;

			const bool zwrite = rendering_attributes.settings.enable_depth_copy;
			
			void* ptr;
			size_t size;

			void* depth_ptr = nullptr;
			size_t depth_size;
			
			static GLuint texid = (GLuint)-1;
			static GLuint depthbufid = (GLuint)-1;
			
			static GLuint pbo = (GLuint)-1;
			static GLuint depth_pbo = (GLuint)-1;
			static GLuint fbo = (GLuint)-1;
			
			static cudaGraphicsResource* glFb = nullptr;
			static cudaGraphicsResource* glDepthFb = nullptr;

			static int32_t width = -1;
			static int32_t height = -1;

			// Initialise
			if (texid == (uint32_t)-1)
			{
				// Create pbos
				glGenBuffers(1, &pbo);
				glGenBuffers(1, &depth_pbo);

				// Create fbo
				glGenFramebuffers(1, &fbo);

				// Create opengl texture for cuda framebuffer
				glGenTextures(1, &texid);

				// Create depth texture
				glGenTextures(1, &depthbufid);
			}

			// Resize buffers if size changed
			if (width != viewport.w || height != viewport.h)
			{
				width = viewport.w;
				height = viewport.h;

				// Colour pixel buffer object
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
				glBufferData(GL_PIXEL_UNPACK_BUFFER,
					viewport.w * viewport.h * 4 * sizeof(GLubyte),
					nullptr, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

				// Depth pixel buffer object
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, depth_pbo);
				glBufferData(GL_PIXEL_UNPACK_BUFFER,
					viewport.w * viewport.h * sizeof(GLfloat),
					nullptr, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

				// Texture
				glBindTexture(GL_TEXTURE_2D, texid);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glBindTexture(GL_TEXTURE_2D, 0);

				// Depth buffer
				float* depth_buffer_data = (float*)malloc(width * height * sizeof(float));
				for (int i = 0; i < width * height; ++i)
				{
					depth_buffer_data[i] = 1.0f;
				}

				glBindTexture(GL_TEXTURE_2D, depthbufid);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buffer_data);
				glBindTexture(GL_TEXTURE_2D, 0);

				free(depth_buffer_data);

				// Register pixel buffer
				if (glFb != nullptr)
					gpuErrchk(cudaGraphicsUnregisterResource(glFb));
				gpuErrchk(cudaGraphicsGLRegisterBuffer(&glFb, pbo, cudaGraphicsRegisterFlagsNone));

				// Register depth buffer
				if (glDepthFb != nullptr)
					gpuErrchk(cudaGraphicsUnregisterResource(glDepthFb));
				gpuErrchk(cudaGraphicsGLRegisterBuffer(&glDepthFb, depth_pbo, cudaGraphicsRegisterFlagsNone));
			}

			// Update rendering attributes
			rendering_attributes.settings.refraction_discrete_steps_max = 1 +
				(int)(1.0f / rendering_attributes.settings.refraction_discrete_step);

			// Bind cuda graphics resources
			gpuErrchk(cudaGraphicsMapResources(1, &glFb, 0));

			// Get a device pointer to it
			gpuErrchk(cudaGraphicsResourceGetMappedPointer(&ptr, &size, glFb));

			// Bind depth resources
			if (zwrite)
			{
				gpuErrchk(cudaGraphicsMapResources(1, &glDepthFb, 0));
				gpuErrchk(cudaGraphicsResourceGetMappedPointer(&depth_ptr, &depth_size, glDepthFb));
			}
			
			// Calculate number of threads per block
			dim3 block_size(32, 32);

			// Calculate grid size from this
			// Block width = ceil(width / grid_size.x) etc
			dim3 grid_size((width + block_size.x - 1) / block_size.x,
				(height + block_size.y - 1) / block_size.y);

			// Execute kernel
			cudaRenderOctree<<<grid_size, block_size>>>(treeGpu, (int32_t*)ptr, (GLfloat*)depth_ptr, rendering_attributes);

			// Check for errors
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaPeekAtLastError());

			// Unmap pbo
			gpuErrchk(cudaGraphicsUnmapResources(1, &glFb, 0));

			if (zwrite)
				gpuErrchk(cudaGraphicsUnmapResources(1, &glDepthFb, 0));
			
			// Render PBO to screen
			// Reset OpenGL matrices
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			// Draw fullscreen quad
			glDisable(GL_LIGHTING);
			glEnable(GL_TEXTURE_2D);

			// Copy colour data from pbo to texture
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(GL_TEXTURE_2D, texid);
			glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

			// Copy depth data from pbo to texture
			if (zwrite)
			{
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, depth_pbo);
				glBindTexture(GL_TEXTURE_2D, depthbufid);
				glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
			}
			
			// Render PBO to screen
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);

			const float vert[] =
			{
				-1.0f, -1.0f,
				-1.0f, 1.0f,
				1.0f, 1.0f,
				1.0f, -1.0f
			};

			const float tex_coord[] =
			{
				0.0f, 0.0f,
				0.0f, 1.0f,
				1.0f, 1.0f,
				1.0f, 0.0f
			};

			glVertexPointer(2, GL_FLOAT, 2 * sizeof(float), vert);
			glTexCoordPointer(2, GL_FLOAT, 2 * sizeof(float), tex_coord);

			glBindTexture(GL_TEXTURE_2D, texid);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_TEXTURE_COORD_ARRAY);
			
			// Write depth buffer
			if (zwrite)
			{
				glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
				glBindTexture(GL_TEXTURE_2D, depthbufid);
				glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthbufid, 0);
				glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
				glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			}

			// Unbind PBO
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			// Restore OpenGL matrices
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();

			glMatrixMode(GL_MODELVIEW);
			glPopMatrix();
		}
	}
}
