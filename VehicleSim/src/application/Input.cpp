#include "VehicleSim.h"

#include "input/InputConverter.h"

#include "serialisation/Serialiser.h"

namespace vlr
{
	void VehicleSim::doFrameInput(float dt)
	{
		// Update tools
		for (auto it = _tools.begin(); it != _tools.end(); ++it)
		{
			(*it)->update_base(dt);
		}
	}

	void VehicleSim::resize(int width, int height)
	{
		_width = getWidth();
		_height = getHeight();
		_aspect = (float)_width / (float)_height;

		float widthOver2 = width / 2.0f;
		float heightOver2 = height / 2.0f;

		_camera.setViewport(widthOver2, 0, widthOver2, heightOver2);
		_camera.orthographic(10, _aspect);

		_guiCanvas->SetSize(width, height);
	}

	void VehicleSim::mouse_callback(GLFWwindow* window, int button,
		int action, int mods)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;
		b2World& physWorld = app->_physWorld;
		common::Camera& cam = app->_camera;

		// Update GUI
		canvas->InputMouseButton(button, (action == GLFW_PRESS));

		// Get viewport
		common::Viewport viewport = cam.getViewport();

		// Update tools
		std::vector<Tool*>& tools = app->_tools;
		for (auto it = tools.begin(); it != tools.end(); ++it)
		{
			(*it)->click_base(button, action, mods);
		}
	}

	void VehicleSim::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;
		common::Camera& cam = app->_camera;

		float height = (float)app->getHeight();

		// Calculate difference
		double oldX = app->_mouseX;
		double oldY = -app->_mouseY + height;
		double dx = x - app->_mouseX;
		double dy = y - app->_mouseY;
		app->_mouseX = x;
		app->_mouseY = y;

		// Update GUI
		canvas->InputMouseMoved((int)x, (int)y, (int)dx, (int)dy);

		// Update tools
		std::vector<Tool*>& tools = app->_tools;
		for (auto it = tools.begin(); it != tools.end(); ++it)
		{
			(*it)->mousemove_base(x, height - y, dx, dy);
		}
	}

	void VehicleSim::key_callback(GLFWwindow* window, int key,
		int scancode, int action, int mods)
	{
		// Do default action (exit on esc)
		_default_key_callback(window, key, scancode, action, mods);

		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;

		// Update GUI
		canvas->InputKey(InputConverter::translateKeyCode(key), action == GLFW_PRESS);

		// Update tools
		std::vector<Tool*>& tools = app->_tools;
		for (auto it = tools.begin(); it != tools.end(); ++it)
		{
			(*it)->key_base(key, scancode, action, mods);
		}
	}
	
	void VehicleSim::scroll_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;

		// Update GUI
		canvas->InputMouseWheel((int)(y * 60.0));

		// Update app
		//app->_orthoScale -= y;
		//if (app->_orthoScale <= VEHICLESIM_MIN_SCALE)
		//	app->_orthoScale = VEHICLESIM_MIN_SCALE;

		// Update tools
		std::vector<Tool*>& tools = app->_tools;
		for (auto it = tools.begin(); it != tools.end(); ++it)
		{
			(*it)->scroll_base(x, y);
		}
	}
	
	void VehicleSim::char_callback(GLFWwindow* window,
			unsigned int codepoint)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;

		// Update GUI
		canvas->InputCharacter(codepoint);
	}
	
	void VehicleSim::resize_callback(GLFWwindow* window,
			int width, int height)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);

		app->resize(width, height);
	}
}
