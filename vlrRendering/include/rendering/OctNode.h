#ifndef VLR_RENDERING_OCTNODE
#define VLR_RENDERING_OCTNODE

#include <glm/vec3.hpp>
#include <string.h>

namespace vlr
{
	namespace rendering
	{
		class OctNode
		{
		public:
			__host__ __device__ OctNode()	
				: leaf(true)
			{
				memset(children, 0, 8 * sizeof(OctNode*));
			}     

			__host__ __device__ ~OctNode()
			{
				for (int i = 0; i < 8; ++i)
				{
					if (far[i])
						delete children[i];
				}
			}

			bool leaf;
			bool far[8];
			OctNode* children[8];
			float4 normal;
		};
	}
}

#endif /* VLR_RENDERING_OCTNODE */
