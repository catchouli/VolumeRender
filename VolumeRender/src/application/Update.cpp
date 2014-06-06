#include "application/VolumeRender.h"

namespace vlr
{
	void VolumeRender::update(double dt)
	{
		const float MOVE_SPEED = 2.0f;

		// Count frames and output after 10 seconds
		static int frames = 0;
		frames++;

		// Calculate average FPS
		static bool done = false;
		static double start_time = glfwGetTime();
		double time = glfwGetTime();

		if (!done && (time - start_time) >= 10.0)
		{
			done = true;

			double avg_fps = (double)frames / (time - start_time);

			printf("Average FPS: %f\n", avg_fps);
		}

		// Set window title
		const int32_t TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, "FPS: %d\n, [2] Toggle shadows (%s), [3] Toggle reflection (%s), [4] Toggle refraction (%s)", getFPS(),
							(rendering_attributes.settings.enable_shadows ? "on" : "off"),
							(rendering_attributes.settings.enable_reflection ? "on" : "off"),
							(rendering_attributes.settings.enable_refraction ? "on" : "off"));
		glfwSetWindowTitle(_window, title);

		if (glfwGetKey(_window, GLFW_KEY_S))
			printf("Cam pos: %f %f %f, Cam rot: %f %f\n",
					_camera.getPos().x,  _camera.getPos().y, _camera.getPos().z,
					_camRot.x, _camRot.y);

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
