#ifndef VLR_RENDERING_COLOUR_H
#define VLR_RENDERING_COLOUR_H

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

namespace vlr
{
	namespace rendering
	{
		// Helper struct
		struct Colour
		{
			uint32_t r : 8;
			uint32_t g : 8;
			uint32_t b : 8;
			uint32_t a : 8;
		};
		
		__host__ __device__ uint32_t compressColour(glm::vec4 colour);
		__host__ __device__ glm::vec4 decompressColour(uint32_t colour);
	}
}

#endif /* VLR_RENDERING_COLOUR_H */
