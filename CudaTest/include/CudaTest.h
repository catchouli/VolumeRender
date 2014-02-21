#ifndef VLR_RAY2D_APPLICATION
#define VLR_RAY2D_APPLICATION

#include "app/Application.h"
#include "rendering/Camera.h"

#include <glm/glm.hpp>

#include <stdio.h>

namespace vlr
{
	const int RAY2D_GRID_WIDTH = 8;
	const int RAY2D_GRID_HEIGHT = 8;
	const int RAY2D_GRID_DEPTH = 8;

	class CudaTest
	: public common::Application
	{
	public:
		CudaTest();

		void update(double dt);

		void render();

		void genGrid();

		// Callbacks
		static void mouse_move_callback(GLFWwindow* window,
			double x, double y);
		static void mouse_callback(GLFWwindow* window, int button,
			int action, int mods);
		static void key_callback(GLFWwindow* window, int key,
			int scancode, int action, int mods);

	protected:

	private:
		common::Camera _camera;

		int _grid[RAY2D_GRID_WIDTH * RAY2D_GRID_HEIGHT * RAY2D_GRID_DEPTH];
		
		glm::mat4 _mvp;

		double _mouseX, _mouseY;

		int _width, _height;

		GLuint _texid;
		GLuint _pbo;

		cudaGraphicsResource* _glFb;

		// GPU resources
		int* _gridGpu;
		glm::mat4* _mvpGpu;
		glm::vec3* _originGpu;
	};

}

#endif /* VLR_RAY2D_APPLICATION */
