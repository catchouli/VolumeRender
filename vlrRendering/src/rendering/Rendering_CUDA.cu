#include "rendering/Rendering.h"

#include "rendering/Shading.h"
#include "rendering/rendering_attributes.h"
#include "resources/Image.h"
#include "util/Util.h"
#include "util/CUDAUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include <stdint.h>

namespace vlr
{
	namespace rendering
	{
		__device__ __host__ inline ray screenPointToRay(int32_t x, int32_t y,
			const rendering_attributes_t rendering_attributes)
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

			ret.direction = glm::vec3(viewportPos * rendering_attributes.mvp);

			return ret;
		}

		__global__ void cudaRenderOctree(const int32_t* root, int32_t* pixel_buffer,
			const rendering_attributes_t rendering_attributes)
		{
			const viewport& viewport = rendering_attributes.viewport;

			const int32_t width = viewport.w;		// Viewport width
			const int32_t height = viewport.h;		// Viewport height
			
			ray ray;							// Eye ray for this kernel
			int32_t x, y;							// x and y of this kernel's pixel
			int32_t* pixel;							// Pointer to this kernel's pixel

			float hit_t;						// Resultant hit t value from raycast
			glm::vec3 hit_pos;					// Resultant hit position from raycast
			const int32_t* hit_parent;				// Resultant hit parent voxel from raycast
			int32_t hit_idx;						// Resultant hit child index from raycast
			int32_t hit_scale;						// Resultant hit scale from raycast

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
			ray = screenPointToRay(x, y, rendering_attributes);

			// Clear to black
			*pixel = 0;
			
			// Skip first child desc slot (it's a pointer to the info section)
			const int32_t* root_child_desc = root + child_desc_size_ints;

			// Do raycast
			raycast(root_child_desc, &ray, &hit_t, &hit_pos, &hit_parent, &hit_idx, &hit_scale);

			// If we hit a voxel in the tree
			if (hit_scale < MAX_SCALE)
			{
				*pixel = shade(rendering_attributes, hit_t, hit_pos, root, hit_parent, hit_idx, hit_scale);
			}
		}

		void renderOctree(const int32_t* treeGpu, const rendering_attributes_t rendering_attributes)
		{
			const char* blit_pixel_shader = "blit_buffer.ps";

			const viewport& viewport = rendering_attributes.viewport;

			void* ptr;
			size_t size;

			static GLuint texid = (GLuint)-1;
			static GLuint pbo = (GLuint)-1;
			static cudaGraphicsResource* glFb = nullptr;
			
			static GLuint pixel_shader = (GLuint)-1;
			static GLuint shader_program = (GLuint)-1;

			static int32_t width = -1;
			static int32_t height = -1;

			// Initialise
			if (texid == (uint32_t)-1)
			{
				// Create pbo
				glGenBuffers(1, &pbo);

				// Create opengl texture for cuda framebuffer
				glGenTextures(1, &texid);

				// Create pixel shader
				pixel_shader = glCreateShader(GL_FRAGMENT_SHADER);

				// Load pixel shader source
				char* pixel_shader_source;
				GLint len = (GLint)read_full_file(blit_pixel_shader, &pixel_shader_source);

				if (pixel_shader_source == nullptr)
				{
					fprintf(stderr, "Failed to load fragment shader: %s\n", blit_pixel_shader);
					//exit(EXIT_FAILURE);
				}

				// Compile shader
				glShaderSource(pixel_shader, 1, (const GLchar**)&pixel_shader_source, &len);
				glCompileShader(pixel_shader);

				// Free pixel shader source memory
				free(pixel_shader_source);

				// Create shader program
				shader_program = glCreateProgram();
				glAttachShader(shader_program, pixel_shader);
			}

			// Resize buffers if size changed
			if (width != viewport.w || height != viewport.h)
			{
				width = viewport.w;
				height = viewport.h;

				// pixel_buffer object
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
				glBufferData(GL_PIXEL_UNPACK_BUFFER,
					viewport.w * viewport.h * 4 * sizeof(GLubyte),
					nullptr, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

				// Texture
				glBindTexture(GL_TEXTURE_2D, texid);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR/*or GL_NEAREST*/);
				glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR/*or GL_NEAREST*/);
				glBindTexture(GL_TEXTURE_2D, 0);

				// Register buffer
				if (glFb != nullptr)
					gpuErrchk(cudaGraphicsUnregisterResource(glFb));
				gpuErrchk(cudaGraphicsGLRegisterBuffer(&glFb, pbo, cudaGraphicsRegisterFlagsNone));
			}

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
			cudaRenderOctree<<<grid_size, block_size>>>(treeGpu, (int32_t*)ptr, rendering_attributes);

			// Check for errors
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaPeekAtLastError());

			// Unmap pbo
			gpuErrchk(cudaGraphicsUnmapResources(1, &glFb, 0));
			
			// Render PBO to screen
			// Reset OpenGL matrices
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			// Enable shader
			//glUseProgram(shader_program);

			// Draw fullscreen quad
			glDisable(GL_LIGHTING);
			glEnable(GL_TEXTURE_2D);

			// Copy data from pbo to texture
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(GL_TEXTURE_2D, texid);
			glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			
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

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_TEXTURE_COORD_ARRAY);

			// Disable shader
			glUseProgram(0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			// Restore OpenGL matrices
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();

			glMatrixMode(GL_MODELVIEW);
			glPopMatrix();
		}
	}
}
