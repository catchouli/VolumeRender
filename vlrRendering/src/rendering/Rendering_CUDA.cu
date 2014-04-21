#include "rendering/Rendering.h"

#include "util/CUDAUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

namespace vlr
{
	namespace rendering
	{
		__global__ void cudaRenderOctree(const Octree* tree, int* pixel_buffer, const rendering::float4* origin,
			const mat4* mvp, const viewport* viewport)
		{
			const int width = viewport->w;		// Viewport width
			const int height = viewport->h;		// Viewport height
			
			ray ray;							// Eye ray for this kernel
			int x, y;							// x and y of this kernel's pixel
			int* pixel;							// Pointer to this kernel's pixel

			float hit_t;							// Resultant hit t value from raycast
			float3 hit_pos;						// Resultant hit position from raycast
			OctNode* hit_parent;					// Resultant hit parent voxel from raycast
			int hit_idx;							// Resultant hit child index from raycast
			int hit_scale;						// Resultant hit scale from raycast

			// Calculate x and y
			// (each block works on a blockDim.x * blockDim.y square of pixels, so
			// p = block index * block dimensions + thread index;
			x = blockIdx.x * blockDim.x + threadIdx.x;
			y = blockIdx.y * blockDim.y + threadIdx.y;

			// Due to the const block dimensions, some pixels outside of the screen
			// may have kernels spawned, but no < 0 values are possible
			if (x >= width || y >= height)
				return;

			// Get the pixel in the pixel buffer using the x and y coordinate
			pixel = pixel_buffer + width * y + x;

			// Calculate eye ray for pixel
			screenPointToRay(x, y, origin, mvp, viewport, &ray);

			// Clear to black
			*pixel = 0;

			// Do raycast
			raycast(tree, &ray, &hit_t, &hit_pos, &hit_parent, &hit_idx, &hit_scale);

			// Shade
			if (hit_scale < MAX_SCALE)
			{
				OctNode* child = hit_parent + (int)hit_parent->children[hit_idx];

				int r = (int)(child->normal.x * 255.0f);
				int g = (int)(child->normal.y * 255.0f);
				int b = (int)(child->normal.z * 255.0f);

				*pixel = (r) + (g << 8) + (b << 16);
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

			static GLuint texid = (unsigned int)-1;
			static GLuint pbo = (unsigned int)-1;
			static cudaGraphicsResource* glFb = nullptr;

			static int width = -1;
			static int height = -1;

			// Initialise
			if (texid == (unsigned int)-1)
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

				// pixel_buffer object
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
			cudaRenderOctree<<<grid_size, block_size>>>(octreeGpu, (int*)ptr, originGpu, mvpGpu, viewportGpu);

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
	}
}
