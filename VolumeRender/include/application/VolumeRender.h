#ifndef VLR_VOLUMERENDER_APPLICATION
#define VLR_VOLUMERENDER_APPLICATION

#include "app/Application.h"
#include "app/Framebuffer.h"
#include "maths/Types.h"
#include "rendering/Camera.h"
#include "rendering/Octree.h"
#include "resources/Mesh.h"

#include <stdio.h>

namespace vlr
{
	const int VolumeRender_GRID_WIDTH = 8;
	const int VolumeRender_GRID_HEIGHT = 8;
	const int VolumeRender_GRID_DEPTH = 8;
	
	class VolumeRender
		: public rendering::Application
	{
	public:
		VolumeRender(int argc, char** argv);

		void update(double dt);

		void render();

		void genGrid();

		// Callbacks
		static void resize_callback(GLFWwindow* window,
			int width, int height);
		static void mouse_move_callback(GLFWwindow* window,
			double x, double y);
		static void mouse_callback(GLFWwindow* window, int button,
			int action, int mods);
		static void key_callback(GLFWwindow* window, int key,
			int scancode, int action, int mods);

	private:
		const char* _treeFilename;

		double _mouseX, _mouseY;

		rendering::Mesh _mesh;
		
		glm::vec3 normal;
		int lastx, lasty, lastz;
		
		rendering::Camera _camera;

		int* _gpuTree;
		const int* itree;

		rendering::Framebuffer _fb;

		float _rot;

		bool _cursorLocked;

		glm::vec3 _camRot;
	};
}

#endif /* VLR_VOLUMERENDER_APPLICATION */
