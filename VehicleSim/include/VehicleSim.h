#ifndef VLR_RAY2D_APPLICATION
#define VLR_RAY2D_APPLICATION

#include "app/Application.h"
#include "app/Framebuffer.h"
#include "rendering/Camera.h"
#include "rendering/Mesh.h"
#include "rendering/Octree.h"
#include "rendering/OctNode.h"

#include <stdio.h>

namespace vlr
{
	const int RAY2D_GRID_WIDTH = 8;
	const int RAY2D_GRID_HEIGHT = 8;
	const int RAY2D_GRID_DEPTH = 8;

	class Ray2D
	: public common::Application
	{
	public:
		Ray2D();

		void update(double dt);

		void render();

		void genGrid();
		void genOctree(common::Octree& tree);
		void genNode(common::OctNode** node, glm::vec3 min, glm::vec3 max, int depth, int maxDepth);
		void renderOctreeGL(common::Octree tree);
		void renderNodeGL(common::OctNode* node, glm::vec3 min, glm::vec3 max);

		bool raycastScreenPointGrid(int x, int y);
		bool raycastGrid(glm::vec3 origin, glm::vec3 direction);

		bool raycastScreenPointOctree(int x, int y);
		bool raycastOctree(glm::vec3 origin, glm::vec3 direction);

		// Callbacks
		static void mouse_move_callback(GLFWwindow* window,
			double x, double y);
		static void mouse_callback(GLFWwindow* window, int button,
			int action, int mods);
		static void key_callback(GLFWwindow* window, int key,
			int scancode, int action, int mods);

	protected:
		struct Ray
		{
			glm::vec3 origin;
			glm::vec3 direction;
		};

		Ray screenPointToRay(int x, int y);

	private:
		double _mouseX, _mouseY;
		
		glm::vec3 normal;
		int lastx, lasty, lastz;
		
		common::Camera _camera;

		int _grid[RAY2D_GRID_WIDTH][RAY2D_GRID_HEIGHT][RAY2D_GRID_DEPTH];

		common::Octree _tree;

		common::Framebuffer _fb;

		common::Mesh mesh;

		float _rot;

		bool _cursorLocked;

		glm::vec3 unprojected;

		glm::vec3 lastraypos;
		glm::vec3 lastraydir;
	};

}

#endif /* VLR_RAY2D_APPLICATION */
