#include "application/Ray2D.h"

namespace vlr
{
	void Ray2D::renderOctreeGL(rendering::Octree tree)
	{
		renderNodeGL(tree.root, tree.min, tree.max);
	}

	void Ray2D::renderNodeGL(rendering::OctNode* node, glm::vec3 min,
		glm::vec3 max)
	{
		if (node->leaf)
		{
			glm::vec3 scale = max - min;
			
			glLoadIdentity();
			glTranslatef(min.x, min.y, min.z);
			glScalef(scale.x, scale.y, scale.z);
			mesh.render();
			return;
		}

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

					// Get node
					rendering::OctNode* newNode = node->children[idx];
					if (newNode != nullptr)
						renderNodeGL(newNode, newMin, newMax);

				}
			}
		}
	}
}
