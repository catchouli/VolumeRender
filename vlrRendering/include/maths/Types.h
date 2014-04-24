#ifndef VLR_RENDERING_TYPES
#define VLR_RENDERING_TYPES

#include "Matrix.h"
#include <glm/glm.hpp>

namespace vlr
{
	namespace rendering
	{
		union colour
		{
			struct
			{
				unsigned char r, g, b, a;
			};

			int col;
		};

		struct ray
		{
			glm::vec3 origin;
			glm::vec3 direction;
		};
	}
}

#endif /* VLR_RENDERING_TYPES */