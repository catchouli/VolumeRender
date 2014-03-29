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

		// Make noncontiguous tree
		rendering::OctNode* root = new rendering::OctNode();
		
		// Calculate position
		const glm::vec3 pos(0, 0, 10);

		glm::vec3 min(0, 0, 0);
		glm::vec3 max(RAY2D_GRID_WIDTH, RAY2D_GRID_HEIGHT,
			RAY2D_GRID_DEPTH);

		// Generate nodes
		genNode(&root, min, max, 0, MAX_DEPTH);

		// Make tree contiguous
		genContiguousTree(tree, root);
		tree.min = min;
		tree.max = max;
		tree.depth = MAX_DEPTH;

		// Clean up old nodes
		delete root;
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

					// Calculate index of children (low order bit is x, second bit is y, third bit is z dir)
					int idx = (z << 2) | (y << 1) | x;

					rendering::OctNode** nextChild = &((*node)->children[idx]);
					(*node)->far[idx] = true;
					
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

	int nodeCount(rendering::OctNode* node)
	{
		if (node == nullptr)
			return 0;

		int nodes = 1;

		for (int i = 0; i < 8; ++i)
		{
			nodes += nodeCount(node->children[i]);
		}

		return nodes;
	}

	rendering::OctNode* copyToArray(rendering::OctNode* node, rendering::OctNode* array, int& current)
	{
		if (node == 0)
			return (rendering::OctNode*)0;

		int baseNodeId = current;
		rendering::OctNode* currentNode = &array[current++];

		currentNode->leaf = node->leaf;

		for (int i = 0; i < 8; ++i)
		{
			// Copy children and store relative pointers
			rendering::OctNode* ptr = copyToArray(node->children[i], array, current);

			// Copy pointer
			currentNode->children[i] = ptr;

			// Convert to relative pointer if not null
			if (ptr != nullptr)
			{
				// Convert pointer to relative pointer
				currentNode->far[i] = false;
				currentNode->children[i] = (rendering::OctNode*)(ptr - currentNode);
			}
		}

		return currentNode;
	}

	void Ray2D::genContiguousTree(rendering::Octree& tree, rendering::OctNode* root)
	{
		int currentNode = 0;

		// Recursively count nodes
		int allNodes = nodeCount(root);

		tree.root = new rendering::OctNode[allNodes];
		tree.nodeCount = allNodes;

		rendering::OctNode* array = copyToArray(root, tree.root, currentNode);

		// Check I didn't mess this up
		assert(tree.root == array);
		assert(currentNode == allNodes);
	}
}
