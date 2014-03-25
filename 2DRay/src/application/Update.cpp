#include "application/Ray2D.h"

namespace vlr
{
	void Ray2D::update(double dt)
	{
		const float MOVE_SPEED = 2.0f;

		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, "FPS: %d\n", getFPS());
		glfwSetWindowTitle(_window, title);

		// Rotate cube
		_rot += (float)dt * 100.0f;

		// Handle movement
		if (_cursorLocked)
		{
			if (glfwGetKey(_window, GLFW_KEY_W))
			{
				_camera.translate(MOVE_SPEED * (float)dt * _camera.getForward());
			}
			if (glfwGetKey(_window, GLFW_KEY_S))
			{
				_camera.translate(-MOVE_SPEED * (float)dt * _camera.getForward());
			}
			if (glfwGetKey(_window, GLFW_KEY_A))
			{
				_camera.translate(MOVE_SPEED * (float)dt * _camera.getLeft());
			}
			if (glfwGetKey(_window, GLFW_KEY_D))
			{
				_camera.translate(-MOVE_SPEED * (float)dt * _camera.getLeft());
			}
			if (glfwGetKey(_window, GLFW_KEY_Q))
			{
				_camera.translate(glm::vec3(0, -MOVE_SPEED * (float)dt, 0));
			}
			if (glfwGetKey(_window, GLFW_KEY_E))
			{
				_camera.translate(glm::vec3(0, MOVE_SPEED * (float)dt, 0));
			}
		}
	}
}
