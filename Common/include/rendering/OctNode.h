#ifndef VLR_COMMON_OCTNODE
#define VLR_COMMON_OCTNODE

#include <glm/vec3.hpp>
#include <string.h>

namespace vlr
{
	namespace common
	{
		class OctNode
		{
		public:
			OctNode()	
				: leaf(true)
			{
				memset(children, 0, 8 * sizeof(OctNode*));
			}     

			bool leaf;
			OctNode* children[8];
		};
	}
}

#endif /* VLR_COMMON_OCTNODE */
