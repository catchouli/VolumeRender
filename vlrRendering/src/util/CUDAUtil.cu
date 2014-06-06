#include "util/CudaUtil.h"

#include <assert.h>
#include <GL/glew.h>

#include <assimp/scene.h>
#include <assimp/postprocess.h>

void gpuAssert(cudaError_t code, char *file, int32_t line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

		system("pause");

		if (abort)
			exit(code);
	}
}

namespace vlr
{
	namespace rendering
	{
			// A table for getting the child index for the child child_index
			// the index in this array = (parent's_childmask << child index)
			__constant__ int32_t child_index_table[] =
			{
				0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
				1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
				1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
				2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
				1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
				2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
				2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
				3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
				1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
				2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
				2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
				3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
				2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
				3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
				3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
				4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
			};

		__device__ int32_t get_child_index(uint32_t mask)
		{
			return child_index_table[mask & 0xFFu];
		}

		// Get number of set bits
		// From http://stackoverflow.com/a/109025
		// (you are not meant to understand this)
		__host__ __device__ int numberOfSetBits(int i)
		{
			 i = i - ((i >> 1) & 0x55555555);
			 i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
			 return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
		}

		void checkGlError()
		{
			GLuint gl_error = glGetError();

			if (gl_error != GL_NO_ERROR)
			{
				fprintf(stderr, "OpenGL error: %s\n", gluErrorString(gl_error));

				assert(gl_error == GL_NO_ERROR);
			}
		}

		void renderAiScene(const aiScene* scene)
		{
		}
	}
}
