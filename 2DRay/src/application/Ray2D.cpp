#include "application/Ray2D.h"

#include "util/CUDAUtil.h"
#include "rendering/Raycast.h"

#include "rendering/OctNode.h"

namespace vlr
{
	Ray2D::Ray2D()
		: Application(513, 512), _rot(0)
	{
		// Set callbacks
		glfwSetFramebufferSizeCallback(_window, resize_callback);
 		glfwSetCursorPosCallback(_window, mouse_move_callback);
		glfwSetMouseButtonCallback(_window, mouse_callback);
		glfwSetKeyCallback(_window, key_callback);

		// Lock cursor
		_cursorLocked = true;
		glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		// Get cursor pos
		glfwGetCursorPos(_window, &_mouseX, &_mouseY);

		// Initialise framebuffer
		_fb.resize(getWidth(), getHeight());

		// Generate cube
		mesh.createCube();

		// Initialise camera
		int width = getWidth();
		int height = getHeight();
		float aspect = (float)height / (float)width;

		// Set camera rotation
		_camRot = glm::vec3(0.6f, -0.65f, 0);

		_camera.setViewport(0, 0, width, height);
		_camera.perspective((float)(3.14159265358 / 2.0), aspect, 0.01f, 100.0f);
		_camera.translate(glm::vec3(2.7f, 2.8f, 2.9f));

		// Rotate camera to initial rotation
		_camera.rotate(glm::vec3(_camRot.x, 0, 0));
		_camera.rotate(glm::vec3(0, _camRot.y, 0));
		_camera.rotate(glm::vec3(0, 0, _camRot.z));

		// Generate grid
		genGrid();

		// Generate octree from grid
		//genOctreeGrid(_tree, &_grid[0][0][0], glm::vec3(RAY2D_GRID_WIDTH, RAY2D_GRID_HEIGHT, RAY2D_GRID_DEPTH));

		// Generate octree from sphere
		genOctreeSphere(_tree, glm::vec3(0.5f, 0.5f, 0.5f), 0.5f);

		// Initialise pointers
		_cpuTree = &_tree;

#ifndef VLR_RAYCAST_CPU
		// Initialise GPU tree pointer and copy data
		_gpuTree = uploadOctreeCuda(*_cpuTree);
		_currentTree = _gpuTree;
#endif
	}
}
