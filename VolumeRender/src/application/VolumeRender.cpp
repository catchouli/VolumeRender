#include "application/VolumeRender.h"

#include "util/Util.h"
#include "util/CUDAUtil.h"

#include <glm/gtx/transform.hpp>

namespace vlr
{
	VolumeRender::VolumeRender(int32_t argc, char** argv)
		: Application(512, 512), _rot(0), _mesh(true)
	{
		// Get tree filename
		_treeFilename = "miku.tree";

		if (argc > 1)
			_treeFilename = argv[1];

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
		int32_t width = getWidth();
		int32_t height = getHeight();
		float aspect = (float)height / (float)width;

		// Set camera rotation
		_camRot = glm::vec3(0.6f, -0.65f, 0);

		_camera.setViewport(0, 0, width, height);
		_camera.perspective((float)(3.14159265358 / 2.0), aspect, 0.01f, 100.0f);
		_camera.translate(glm::vec3(2.7f, 2.8f, 2.9f));
		//_camera.translate(glm::vec3(1.5f, 1.5f, 5.0f));

		// Rotate camera to initial rotation
		_camera.rotate(glm::vec3(_camRot.x, 0, 0));
		_camera.rotate(glm::vec3(0, _camRot.y, 0));
		_camera.rotate(glm::vec3(0, 0, _camRot.z));

		// Clear rendering attributes
		memset(&rendering_attributes, 0, sizeof(rendering::rendering_attributes_t));
		
		// Load tree
		int32_t tree_size = 0;
		char* tree_data = 0;
		
		tree_size = rendering::read_full_file_binary(_treeFilename, &tree_data);

		//if (tree_size == 0)
		//{
		//	fprintf(stderr, "Invalid tree file: %s\n", _treeFilename);
		//	exit(1);
		//}

		//tree_size =
		//	rendering::genOctreeSphere((int32_t**)&tree_data, 7,
		//	glm::vec3(0.5f, 0.5f, 0.5f), 0.25f);

		//if (_mesh.load("miku.md2"))
		//{
		//	glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
		//	rotation = glm::rotate(rotation, 90.0f, glm::vec3(0, 1.0f, 0));
		//	_mesh.transform(rotation);

		//	double start_time = glfwGetTime();

		//	tree_size =
		//		rendering::genOctreeMesh((int32_t**)&tree_data, 8,
		//		&_mesh);

		//	double end_time = glfwGetTime();

		//	double dt = end_time - start_time;

		//	printf("Time to generate tree: %f\n", dt);
		//}

		printf("%.2fMB\n", tree_size / (1024.0f * 1024.0f));

		// Upload sphere to GPU
		gpuErrchk(cudaMalloc((void**)&_gpuTree, tree_size));
		gpuErrchk(cudaMemcpy(_gpuTree, tree_data, tree_size, cudaMemcpyHostToDevice));

		// Free CPU memory
		free(tree_data);
	}
}
