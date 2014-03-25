#include "application/Ray2D.h"

namespace vlr
{
	void Ray2D::genGrid()
	{
		// Initialise grid with random data
		long long length = RAY2D_GRID_WIDTH * RAY2D_GRID_HEIGHT * RAY2D_GRID_DEPTH;
		
		for (long long i = 0; i < length; ++i)
		{
			((int*)_grid)[i] = rand() % 10 < 3;
		}
	}

	void Ray2D::genOctree(rendering::Octree& tree)
	{
		const int MAX_DEPTH = 3;

		tree.root = new rendering::OctNode();
		tree.min = glm::vec3();
		tree.max = glm::vec3(RAY2D_GRID_WIDTH, RAY2D_GRID_HEIGHT,
			RAY2D_GRID_DEPTH);
		tree.depth = MAX_DEPTH;

		genNode(&tree.root, tree.min, tree.max, 0, MAX_DEPTH);
	}

	void Ray2D::genNode(rendering::OctNode** node, glm::vec3 min, glm::vec3 max, int depth, int maxDepth)
	{
#define ARR_IDX(x, y, z, width, height) x * width * height + y * width + z

		if (depth > maxDepth)
			return;

		// Check if this node contains anything
		for (int x = (int)min.x; x < (int)max.x; ++x)
		{
			for (int y = (int)min.y; y < (int)max.y; ++y)
			{
				for (int z = (int)min.z; z < (int)max.z; ++z)
				{
					if (_grid[x][y][z])
					{
						(*node) = new rendering::OctNode();

						break;
					}
				}

				if (*node != nullptr)
					break;
			}

			if (*node != nullptr)
				break;
		}

		if (*node == nullptr)
			return;

		// Create child nodes
		glm::vec3 halfwidth = 0.5f * glm::vec3(max.x - min.x, 0, 0);
		glm::vec3 halfheight = 0.5f * glm::vec3(0, max.y - min.y, 0);
		glm::vec3 halfdepth = 0.5f * glm::vec3(0, 0, max.z - min.z);

		for (int x = 0; x < 2; ++x)
		{
			for (int y = 0; y < 2; ++y)
			{
				for (int z = 0; z < 2; ++z)
				{
					// Calculate new bounding box
					glm::vec3 newMin = min + (float)x * halfwidth +
						(float)y * halfheight + (float)z * halfdepth;
					glm::vec3 newMax = newMin + halfwidth + halfheight
						+ halfdepth;

					int idx = x*4 + y*2 + z;

					rendering::OctNode** nextChild = &((*node)->children[idx]);
					
					genNode(nextChild, newMin, newMax, depth+1, maxDepth);

					int i = 0;
				}
			}
		}
		
		(*node)->leaf = true;
		for (int i = 0; i < 8; ++i)
		{
			if ((*node)->children[i] != nullptr)
			{
				(*node)->leaf = false;
				return;
			}
		}
	}
}
