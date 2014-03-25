#include "application/Ray2D.h"

namespace vlr
{
	void Ray2D::resize_callback(GLFWwindow* window, int width, int height)
	{
		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = ray2d->_camera;

		cam.setViewport(0, 0, width, height);

		float aspect = (float)height / (float)width;
		cam.perspective(cam.getFov(), aspect, cam.getNear(), cam.getFar());
	}

	void Ray2D::mouse_callback(GLFWwindow* window, int button,
		int action, int mods)
	{
		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = ray2d->_camera;
	}

	void Ray2D::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = ray2d->_camera;

		// Calculate difference
		double diffX = x - ray2d->_mouseX;
		double diffY = y - ray2d->_mouseY;
		
		// Update mouse pos
		ray2d->_mouseX = x;
		ray2d->_mouseY = y;

		// Camera rotation x and y
		static float camRotX = 0;
		static float camRotY = 0;

		if (ray2d->_cursorLocked)
		{
			// Update camera rotation
			camRotX += (float)diffY * 0.001f;
			camRotY += (float)diffX * 0.001f;

			// Rotate camera with mouse movement
			// Calculate rotation
			cam.setRot(glm::quat());
			cam.rotate(glm::vec3(camRotX, 0, 0));
			cam.rotate(glm::vec3(0, camRotY, 0));
		}
	}

	void Ray2D::key_callback(GLFWwindow* window, int key,
		int scancode, int action, int mods)
	{
		// Do default action (exit on esc)
		_default_key_callback(window, key, scancode, action, mods);

		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = ray2d->_camera;

		// Regenerate grid
		if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
			ray2d->genGrid();

		// Lock/unlock cursor & disable input
		if (key == GLFW_KEY_L && action == GLFW_PRESS)
		{
			ray2d->_cursorLocked = !ray2d->_cursorLocked;
			if (ray2d->_cursorLocked)
			{
				glfwSetInputMode(ray2d->_window,
					GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else
			{
				glfwSetInputMode(ray2d->_window,
					GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}
}
