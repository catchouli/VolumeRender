#ifndef VLR_RENDERING_TYPES
#define VLR_RENDERING_TYPES

#include "Matrix.h"

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
			float4 origin;
			float4 direction;
		};
	}
}

#endif /* VLR_RENDERING_TYPES */