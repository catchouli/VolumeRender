#include "maths/Colour.h"

#include "maths/Functions.h"

#include <cuda.h>
#include <math_functions.h>

namespace vlr
{
	namespace rendering
	{
		__host__ __device__ uint32_t compressColour(glm::vec4 colour)
		{
			Colour col;

			glm::vec4 colour255 = glm::clamp(255.0f * colour, 0.0f, 255.0f);
			
			col.r = (uint32_t)colour255.r;
			col.g = (uint32_t)colour255.g;
			col.b = (uint32_t)colour255.b;
			col.a = (uint32_t)colour255.a;

			return *(uint32_t*)&col;
		}

		__host__ __device__ glm::vec4 decompressColour(uint32_t colour)
		{
			Colour col = *(Colour*)&colour;

			return (1.0f / 255.0f) * glm::vec4(col.r, col.g, col.b, col.a);
		}
	}
}
