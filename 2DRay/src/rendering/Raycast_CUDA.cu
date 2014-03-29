#include "rendering/Raycast.h"

#ifndef VLR_RAYCAST_CPU

#define __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "maths/Matrix.h"
#include "maths/Types.h"
#include "rendering/Octree.h"
#include "rendering/Octnode.h"
#include "util/Util.h"
#include "util/CUDAUtil.h"

namespace vlr
{
	namespace rendering
	{
		// Number of threads per block in each dimension (must be a multiple of 32)
		// This is the square of pixels each block fills
		const int THREADS_PER_BLOCK_SQUARE = 32;

		__global__ void raycast(const Octree* tree, int* pixelBuffer, const rendering::float4* origin,
			const mat4* mvp, const viewport* viewport)
		{
			const int s_max = 23;
			const float epsilon = exp2f(-23);

			StackEntry stack[s_max];

			memset(stack, 0, sizeof(StackEntry) * 23);

			int scale = s_max;
			float scale_exp2 = 1.0f;

			float h;
			float t = 0;
			float tmin, tmax;
			int idx;
			float3 pos = { 1.0f, 1.0f, 1.0f };

			const int width = viewport->w;
			const int height = viewport->h;

			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x >= width || y >= height)
				return;

			// Get pixel
			int* pixel = pixelBuffer + width * y + x;

			// Fill with background colour
			*pixel = 0;

			// Calculate eye ray for pixel
			ray ray;
			screenPointToRay(x, y, origin, mvp, viewport, &ray);

			float4 o = ray.origin;
			float4 d = ray.direction;

			// Calculate cube extents for root
			float cubePos[3] = { 1.5f, 1.5f, 1.5f };
			const float cubeScale = 1.0f;
		
			// Eliminate very low (zero) direction values to avoid division by zero
			if (fabs(d.x) < epsilon) d.x = (d.x < 0.0f ? -epsilon : epsilon);
			if (fabs(d.y) < epsilon) d.y = (d.y < 0.0f ? -epsilon : epsilon);
			if (fabs(d.z) < epsilon) d.z = (d.z < 0.0f ? -epsilon : epsilon);
			
			if (d.x > 0)
			{
				d.x = -d.x;
				cubePos[0] = 0.5f;
			}
			if (d.y > 0)
			{
				d.y = -d.y;
				cubePos[1] = 0.5f;
			}
			if (d.z > 0)
			{
				d.z = -d.z;
				cubePos[2] = 0.5f;
			}
		
			float cubeMin[3] = {
				cubePos[0] - 0.5f * cubeScale,
				cubePos[1] - 0.5f * cubeScale,
				cubePos[2] - 0.5f * cubeScale
			};
		
			float cubeMax[3] = {
				cubePos[0] + 0.5f * cubeScale,
				cubePos[1] + 0.5f * cubeScale,
				cubePos[2] + 0.5f * cubeScale
			};

			float cubeCentre[3] = {
				0.5f * (cubeMin[0] + cubeMax[0]),
				0.5f * (cubeMin[1] + cubeMax[1]),
				0.5f * (cubeMin[2] + cubeMax[2]),
			};

			// Precompute ray equation coefficients
			float tx_coef = 1.0f / -(d.x);
			float ty_coef = 1.0f / -(d.y);
			float tz_coef = 1.0f / -(d.z);
		
			float tx_bias = tx_coef * o.x;
			float ty_bias = ty_coef * o.y;
			float tz_bias = tz_coef * o.z;

			// Calculate entry and exit position of ray for each plane
			float txmin = cubeMin[0] * tx_coef + tx_bias;
			float tymin = cubeMin[1] * ty_coef + ty_bias;
			float tzmin = cubeMin[2] * tz_coef + tz_bias;
		
			float txmax = cubeMax[0] * tx_coef + tx_bias;
			float tymax = cubeMax[1] * ty_coef + ty_bias;
			float tzmax = cubeMax[2] * tz_coef + tz_bias;
		
			if (txmin > txmax)
				swap(txmin, txmax);
			if (tymin > tymax)
				swap(tymin, tymax);
			if (tzmin > tzmax)
				swap(tzmin, tzmax);

			// Calculate minimum and maximum t values for root
			float trmin = max(txmin, tymin, tzmin);
			float trmax = min(txmax, tymax, tzmax);
			t = max(t, trmin);
			tmin = t;
			tmax = trmax;
			h = trmax;
		
			// Get first child by evaluating cube at centre
			float txcentre = cubeCentre[0] * tx_coef + tx_bias;
			float tycentre = cubeCentre[1] * ty_coef + ty_bias;
			float tzcentre = cubeCentre[2] * tz_coef + tz_bias;

			// Calculate idx and initial pos
			idx = 0;
			if ((txcentre < trmin) == (d.x >= 0.0f)) { idx ^= 1 << 0; pos.x = 1.5f; }
			if ((tycentre < trmin) == (d.y >= 0.0f)) { idx ^= 1 << 1; pos.y = 1.5f; }
			if ((tzcentre < trmin) == (d.z >= 0.0f)) { idx ^= 1 << 2; pos.z = 1.5f; }

			// Get root node and initialise parent
			OctNode* parent = tree->root;

			// Update scale
			scale = s_max - 1;
			scale_exp2 = 0.5f;		// 2^(scale - s_max)
			
			// Octant rendering (debug)
			if (trmin < trmax)
			{
				const int pink = (int)0x00FF00FF;

				int col = 0;
				
				if (pos.x == 1.5f)
					col |= 0xFF << 0;
				if (pos.y == 1.5f)
					col |= 0xFF << 8;
				if (pos.z == 1.5f)
					col |= 0xFF << 16;

				*pixel = col;
			}
			return;

			// while not terminated
			while (scale < s_max)
			{
				// Get offset of first child (if 0, voxel is not valid)
				int nodeOffset = (int)parent->children[idx];

				// Get pointer to current node
				OctNode* node;
				if (nodeOffset == 0)
					node = 0;
				else
					node = parent + nodeOffset;

				// Project current node
				txmin = pos.x * tx_coef + tx_bias;
				tymin = pos.y * ty_coef + ty_bias;
				tzmin = pos.z * tz_coef + tz_bias;

				txmax = (pos.x + scale_exp2) * tx_coef + tx_bias;
				tymax = (pos.y + scale_exp2) * ty_coef + ty_bias;
				tzmax = (pos.z + scale_exp2) * tz_coef + tz_bias;
		
				if (txmin > txmax)
					swap(txmin, txmax);
				if (tymin > tymax)
					swap(tymin, tymax);
				if (tzmin > tzmax)
					swap(tzmin, tzmax);

				// Calculate minimum and maximum t values for node
				float tcmin = max(txmin, tymin, tzmin);
				float tcmax = min(txmax, tymax, tzmax);

				// If voxel exists && tmin < tmax
				if (node != 0 && tmin < tmax)
				{
					// tv <- intersect(tc, t)
					float tvmin = max(tcmin, trmin);
					float tvmax = min(tcmax, trmax);
					float tv = max(t, tvmin);

					// if tvmin <= tvmax then
					if (tvmin <= tvmax)
					{
						// If voxel is a leaf then return tvmin
						if (node->leaf)
						{
							*pixel = -1;

							return;
						}

						// if tcmax < h then stack[scale] <- (parent, tmax)
						//if (tcmax < h)
						{
							stack[scale].parent = parent;
							stack[scale].tmax = tmax;
						}

						// h <- tmax
						//h = tmax;

						// parent = find child descriptor(parent, idx)
						parent = parent + (int)parent->children[idx];

						// Execute push
						t = tv;

						idx = 0;
						scale -= 1;
						if ((txcentre < tvmin) == (d.x >= 0.0f)) { idx ^= ((d.x >= 0)) << 0; pos.x += exp2f(scale - s_max); }
						if ((tycentre < tvmin) == (d.y >= 0.0f)) { idx ^= ((d.y >= 0)) << 1; pos.y += exp2f(scale - s_max); }
						if ((tzcentre < tvmin) == (d.z >= 0.0f)) { idx ^= ((d.z >= 0)) << 2; pos.z += exp2f(scale - s_max); }

						continue;
					}
				}

				// Advance
				float3 old_pos = pos;

				int step_mask = 0;
				int step_dir = 0;
				if (txmax <= tcmax)
				{
					step_mask ^= 1;

					pos.x += copysign(scale_exp2, d.x);

					if ((idx & (1 << 0)) == 0)
						step_dir ^= (1 << 0);
				}
				if (tymax <= tcmax)
				{
					step_mask ^= 2;
					pos.y += copysign(scale_exp2, d.y);

					if ((idx & (1 << 1)) == 0)
						step_dir ^= (1 << 0);
				}
				if (tzmax <= tcmax)
				{
					step_mask ^= 4;
					pos.z += copysign(scale_exp2, d.z);

					if ((idx & (1 << 2)) == 0)
						step_dir ^= (1 << 0);
				}

				int ray_dir_mask = 0;
				if (d.x > 0) ray_dir_mask |= (1 << 0);
				if (d.x > 0) ray_dir_mask |= (1 << 1);
				if (d.x > 0) ray_dir_mask |= (1 << 2);

				tmin = tcmax;
				idx ^= step_mask;

				// Pop
				// If idx update disagrees with ray then
				if ((idx & step_mask) != 0)
				{
					unsigned int differing_bits = 0;
					if ((step_mask & 1) != 0) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.x + scale_exp2);
					if ((step_mask & 2) != 0) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.y + scale_exp2);
					if ((step_mask & 4) != 0) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.z + scale_exp2);
					float oldScale = scale;
					scale = (__float_as_int((float)differing_bits) >> 23) - 127;
					scale_exp2 = __int_as_float((scale - s_max + 127) << 23);

					if (scale >= s_max)
						return;

					parent = stack[scale].parent;
					tmax = stack[scale].tmax;
					
					int shx = __float_as_int(pos.x) >> scale;
					int shy = __float_as_int(pos.y) >> scale;
					int shz = __float_as_int(pos.z) >> scale;
					pos.x = __int_as_float(shx << scale);
					pos.y = __int_as_float(shy << scale);
					pos.z = __int_as_float(shz << scale);
					idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);

					h = 0.0f;
				}
			}
		}

		void renderOctree(const Octree* octreeGpu, const rendering::float4* origin,
			const mat4* mvp, const viewport* viewport)
		{
			void* ptr;
			size_t size;

			static rendering::float4* originGpu;
			static mat4* mvpGpu;
			static rendering::viewport* viewportGpu;

			static GLuint texid = -1;
			static GLuint pbo = -1;
			static cudaGraphicsResource* glFb = nullptr;

			static int width = -1;
			static int height = -1;

			// Initialise
			if (texid == -1)
			{
				// Create pbo
				glGenBuffers(1, &pbo);

				// Create opengl texture for cuda framebuffer
				glGenTextures(1, &texid);

				// Allocate memory on GPU
				gpuErrchk(cudaMalloc(&originGpu, sizeof(rendering::float4)));
				gpuErrchk(cudaMalloc(&mvpGpu, sizeof(rendering::mat4)));
				gpuErrchk(cudaMalloc(&viewportGpu, sizeof(rendering::viewport)));
			}

			// Resize buffers if size changed
			if (width != viewport->w || height != viewport->h)
			{
				width = viewport->w;
				height = viewport->h;

				// Pixelbuffer object
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
				glBufferData(GL_PIXEL_UNPACK_BUFFER,
					viewport->w * viewport->h * 4 * sizeof(GLubyte),
					nullptr, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

				// Texture
				glBindTexture(GL_TEXTURE_2D, texid);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
				glBindTexture(GL_TEXTURE_2D, 0);

				// Register buffer
				if (glFb != nullptr)
					gpuErrchk(cudaGraphicsUnregisterResource(glFb));
				gpuErrchk(cudaGraphicsGLRegisterBuffer(&glFb, pbo, cudaGraphicsRegisterFlagsNone));
			}

			// Upload data to GPU
			gpuErrchk(cudaMemcpy(originGpu, origin, sizeof(rendering::float4), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(mvpGpu, mvp, sizeof(rendering::mat4), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(viewportGpu, viewport, sizeof(rendering::viewport), cudaMemcpyHostToDevice));

			// Bind cuda graphics resource
			gpuErrchk(cudaGraphicsMapResources(1, &glFb, 0));

			// Get a device pointer to it
			gpuErrchk(cudaGraphicsResourceGetMappedPointer(&ptr, &size, glFb));
			
			// Calculate number of threads per block
			dim3 block_size(THREADS_PER_BLOCK_SQUARE, THREADS_PER_BLOCK_SQUARE);

			// Calculate grid size from this
			// Block width = ceil(width / grid_size.x) etc
			dim3 grid_size((width + block_size.x - 1) / block_size.x,
				(height + block_size.y - 1) / block_size.y);

			// Execute kernel
			raycast<<<grid_size, block_size>>>(octreeGpu, (int*)ptr, originGpu, mvpGpu, viewportGpu);

			// Check for errors
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaPeekAtLastError());

			// Unmap pbo
			gpuErrchk(cudaGraphicsUnmapResources(1, &glFb, 0));

			// Render PBO to screen
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		}

		__global__ void uploadOctreeKernel(Octree* gpuPtr, Octree octree)
		{
			*gpuPtr = octree;
		}

		Octree* uploadOctreeCuda(const Octree& octree)
		{
			Octree gpuOctree = octree;
			Octree* gpuPtr = nullptr;
			OctNode* nodes = nullptr;

			// Only a contiguous tree can be uploaded
			if (octree.nodeCount == 0)
			{
				fprintf(stderr, "Only a contiguous tree can be uploaded\n");

				return gpuPtr;
			}
			
			gpuErrchk(cudaMalloc(&gpuPtr, sizeof(Octree)));
			gpuErrchk(cudaMalloc(&nodes, octree.nodeCount * sizeof(OctNode)));

			gpuOctree.root = nodes;
			gpuErrchk(cudaMemcpy(nodes, octree.root, octree.nodeCount * sizeof(OctNode), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gpuPtr, &gpuOctree, sizeof(Octree), cudaMemcpyHostToDevice));

			return gpuPtr;
		}
	}
}

#endif /* VLR_RAYCAST_CPU */
