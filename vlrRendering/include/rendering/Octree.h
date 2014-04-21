#ifndef VLR_RENDERING_OCTREE
#define VLR_RENDERING_OCTREE

#include "OctNode.h"

#include <glm/vec3.hpp>

namespace vlr
{
	namespace rendering
	{
		struct Octree
		{
			__host__ __device__ Octree()
				: root(nullptr), nodeCount(0)
			{

			}

			__host__ __device__ ~Octree()
			{

			}

			// Depth of the tree
			int depth;

			// Min and max coordinates in world space
			glm::vec3 min, max;

			// The root node
			// Will be an array of all nodes if nodeCount is set
			OctNode* root;

			// The number of nodes in root
			// If 0, root is not an array
			int nodeCount;
		};
	}
}

#endif /* VLR_RENDERING_OCTREE */
