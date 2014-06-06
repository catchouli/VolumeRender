#include "application/VolumeRender.h"

namespace vlr
{
	void VolumeRender::resize_callback(GLFWwindow* window, int32_t width, int32_t height)
	{
		// Get class instance
		VolumeRender* volumeRender = (VolumeRender*)glfwGetWindowUserPointer(window);
		rendering::Camera& cam = volumeRender->_camera;

		if (width == 0 || height == 0)
			return;

		cam.setViewport(0, 0, width, height);

		float aspect = (float)height / (float)width;
		cam.perspective(cam.getFov(), 1.0f / aspect, cam.getNear(), cam.getFar());
	}

	void VolumeRender::mouse_callback(GLFWwindow* window, int32_t button,
		int32_t action, int32_t mods)
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

		static bool first_time = true;
		if (first_time)
		{
			first_time = false;

			// Initialise mouse pos
			glfwGetCursorPos(window, &volumeRender->_mouseX, &volumeRender->_mouseY);
		}

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

	void VolumeRender::key_callback(GLFWwindow* window, int32_t key,
		int32_t scancode, int32_t action, int32_t mods)
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
				glfwGetCursorPos(window, &volumeRender->_mouseX, &volumeRender->_mouseY);

				glfwSetInputMode(volumeRender->_window,
					GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}

		if (key == GLFW_KEY_2 && action == GLFW_PRESS)
			volumeRender->rendering_attributes.settings.enable_shadows = !volumeRender->rendering_attributes.settings.enable_shadows;

		if (key == GLFW_KEY_3 && action == GLFW_PRESS)
			volumeRender->rendering_attributes.settings.enable_reflection = !volumeRender->rendering_attributes.settings.enable_reflection;

		if (key == GLFW_KEY_4 && action == GLFW_PRESS)
			volumeRender->rendering_attributes.settings.enable_refraction = !volumeRender->rendering_attributes.settings.enable_refraction;
	}
}
