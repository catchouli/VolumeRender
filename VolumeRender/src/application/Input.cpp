#include "application/VolumeRender.h"

namespace vlr
{
	void VolumeRender::resize_callback(GLFWwindow* window, int width, int height)
	{
		// Get class instance
		VolumeRender* volumeRender = (VolumeRender*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = volumeRender->_camera;

		if (width == 0 || height == 0)
			return;

		cam.setViewport(0, 0, width, height);

		float aspect = (float)height / (float)width;
		cam.perspective(cam.getFov(), aspect, cam.getNear(), cam.getFar());
	}

	void VolumeRender::mouse_callback(GLFWwindow* window, int button,
		int action, int mods)
	{
		// Get class instance
		VolumeRender* volumeRender = (VolumeRender*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = volumeRender->_camera;
	}

	void VolumeRender::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		VolumeRender* volumeRender = (VolumeRender*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = volumeRender->_camera;

		// Calculate difference
		double diffX = x - volumeRender->_mouseX;
		double diffY = y - volumeRender->_mouseY;
		
		// Update mouse pos
		volumeRender->_mouseX = x;
		volumeRender->_mouseY = y;

		if (volumeRender->_cursorLocked)
		{
			// Update camera rotation
			volumeRender->_camRot.x += (float)diffY * 0.001f;
			volumeRender->_camRot.y += (float)diffX * 0.001f;

			// Rotate camera with mouse movement
			// Calculate rotation
			cam.setRot(glm::quat());
			cam.rotate(glm::vec3(volumeRender->_camRot.x, 0, 0));
			cam.rotate(glm::vec3(0, volumeRender->_camRot.y, 0));
		}
	}

	void VolumeRender::key_callback(GLFWwindow* window, int key,
		int scancode, int action, int mods)
	{
		// Do default action (exit on esc)
		_default_key_callback(window, key, scancode, action, mods);

		// Get class instance
		VolumeRender* volumeRender = (VolumeRender*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = volumeRender->_camera;
		
		// Lock/unlock cursor & disable input
		if (key == GLFW_KEY_L && action == GLFW_PRESS)
		{
			volumeRender->_cursorLocked = !volumeRender->_cursorLocked;
			if (volumeRender->_cursorLocked)
			{
				glfwSetInputMode(volumeRender->_window,
					GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else
			{
				glfwSetInputMode(volumeRender->_window,
					GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}
}
