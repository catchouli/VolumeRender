#include "application/Ray2D.h"

#include "rendering/Raycast.h"
#include "rendering/OctNode.h"
#include "util/CUDAUtil.h"

namespace vlr
{
	Ray2D::Ray2D()
		: Application(512, 512), _rot(0)
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

		// Load mesh
		_mesh.load("miku.md2");

		// Generate sphere
		int* sphere;

		int size =
			rendering::genOctreeSphere(&sphere, 6,
			glm::vec3(5.f, 5.f, 5.f), 5.0f);
		//int size =
		//	rendering::genOctreeMesh(&sphere, 3,
		//	&_mesh);

		printf("%.2fMB\n", size / (1024.0f * 1024.0f));

		// Upload sphere to GPU
		gpuErrchk(cudaMalloc((void**)&_gpuTree, size));
		gpuErrchk(cudaMemcpy(_gpuTree, sphere, size, cudaMemcpyHostToDevice));
	}
}
