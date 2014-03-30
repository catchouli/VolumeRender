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

	void Ray2D::genOctreeSphere(rendering::Octree& tree, glm::vec3 pos, float radius)
	{
		const int MAX_DEPTH = 7;

		// Make noncontiguous tree
		rendering::OctNode* root = new rendering::OctNode();

		glm::vec3 min(0, 0, 0);
		glm::vec3 max(1, 1, 1);

		// Generate nodes
		genNodeSphere(&root, pos, radius, min, max, 0, MAX_DEPTH);

		// Make tree contiguous
		genContiguousTree(tree, root);
		tree.min = min;
		tree.max = max;
		tree.depth = MAX_DEPTH;

		// Clean up old nodes
		delete root;
	}

	// Check whether a box intersects a sphere by calculating the nearest point on
	// any face of the cube to the sphere's centre and then comparing it to r^2
	bool boxSphereIntersection(glm::vec3 min, glm::vec3 max, glm::vec3 pos, float r)
	{
		auto sqr = [](float a) { return a * a; };

		float r_squared = r * r;
		float nearest = 0;

		for(int i = 0; i < 3; i++)
		{
			if(pos[i] < min[i])
				nearest += sqr(pos[i] - min[i]);
			else if(pos[i] > max[i])
				nearest += sqr(pos[i] - max[i]);
		}

		return nearest <= r_squared;
	}

	void Ray2D::genNodeSphere(rendering::OctNode** node, glm::vec3 pos, float radius, glm::vec3 min, glm::vec3 max, int depth, int maxDepth)
	{
		if (depth > maxDepth)
			return;

		// Check if this node contains anything
		if (boxSphereIntersection(min, max, pos, radius))
		{
			(*node) = new rendering::OctNode();
			
		}

		if (*node == nullptr)
			return;

		// Create child nodes
		glm::vec3 halfwidth = 0.5f * glm::vec3(max.x - min.x, 0, 0);
		glm::vec3 halfheight = 0.5f * glm::vec3(0, max.y - min.y, 0);
		glm::vec3 halfdepth = 0.5f * glm::vec3(0, 0, max.z - min.z);

		glm::vec3 centre = halfwidth + halfheight + halfdepth + min;
		glm::vec3 normal = glm::normalize(centre - pos);
 		(*node)->normal.x = normal.x;
		(*node)->normal.y = normal.y;
		(*node)->normal.z = normal.z;

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
					
					genNodeSphere(nextChild, pos, radius, newMin, newMax, depth+1, maxDepth);
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

	void Ray2D::genOctreeGrid(rendering::Octree& tree, int* grid, glm::vec3 size)
	{
		const int MAX_DEPTH = 3;

		// Make noncontiguous tree
		rendering::OctNode* root = new rendering::OctNode();
		
		glm::vec3 min(0, 0, 0);
		glm::vec3 max(RAY2D_GRID_WIDTH, RAY2D_GRID_HEIGHT,
			RAY2D_GRID_DEPTH);

		// Generate nodes
		genNodeGrid(&root, grid, min, max, 0, MAX_DEPTH);

		// Make tree contiguous
		genContiguousTree(tree, root);
		tree.min = min;
		tree.max = max;
		tree.depth = MAX_DEPTH;

		// Clean up old nodes
		delete root;
	}

	void Ray2D::genNodeGrid(rendering::OctNode** node, int* grid, glm::vec3 min, glm::vec3 max, int depth, int maxDepth)
	{
#define ARR_IDX(x, y, z, width, height) x * width * height + y * width + z

		if (depth > maxDepth)
			return;

		// Check if this node contains anything
		for (int x = (int)floor(min.x); x < (int)ceil(max.x); ++x)
		{
			for (int y = (int)floor(min.y); y < (int)ceil(max.y); ++y)
			{
				for (int z = (int)floor(min.z); z < (int)ceil(max.z); ++z)
				{
					int idx = ARR_IDX(x, y, z, (int)max.x, (int)max.y);

					if (grid[idx])
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
					
					genNodeGrid(nextChild, grid, newMin, newMax, depth+1, maxDepth);

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

		memcpy(currentNode, node, sizeof(rendering::OctNode));

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
