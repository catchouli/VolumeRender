#ifndef VLR_COMMON_OCTREE
#define VLR_COMMON_OCTREE

#include "OctNode.h"

#include <glm/vec3.hpp>

namespace vlr
{
	namespace common
	{
		struct StackEntry;

		struct Octree
		{
			~Octree()
			{

			}

			int depth;
			glm::vec3 min, max;
			OctNode* root;
		};
	}
}

#endif /* VLR_COMMON_OCTREE */
