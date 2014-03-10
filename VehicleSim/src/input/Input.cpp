#include "VehicleSim.h"

#include "input/InputConverter.h"

namespace vlr
{
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
		
		// Handle tools
		// If mouse in viewport
		switch (app->_currentTool)
		{
		case Tools::CIRCLE:
			if (viewport.pointInViewport((int)app->_mouseX, (int)app->_mouseY))
			{
				// Create circle body
				if (action == GLFW_PRESS)
				{
					glm::vec3 worldPos = cam.screenSpaceToWorld(app->_mouseX, app->_mouseY, 0);

					printf("%f %f\n", worldPos.x, worldPos.y);

					b2BodyDef bodyDef;
					bodyDef.type = b2_dynamicBody;
					bodyDef.position.Set(worldPos.x, worldPos.y);
					b2Body* body = physWorld.CreateBody(&bodyDef);

					b2CircleShape circle;
					circle.m_p.Set(0.0f, 0.0f);
					circle.m_radius = 1.0f;

					b2FixtureDef fixtureDef;
					fixtureDef.shape = &circle;
					fixtureDef.density = 1.0f;
					fixtureDef.friction = 0.5f;

					body->CreateFixture(&fixtureDef);
				}
			}
		default:
			break;
		}
	}

	void VehicleSim::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance

		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;
		common::Camera& cam = app->_camera;

		// Calculate difference
		double oldX = app->_mouseX;
		double oldY = -app->_mouseY + app->getHeight();
		double dx = x - app->_mouseX;
		double dy = y - app->_mouseY;

		// Calculate world pos
		glm::vec3 oldWorldPos = cam.screenSpaceToWorld(oldX, oldY, 0);
		glm::vec3 newWorldPos = cam.screenSpaceToWorld(x, y, 0);
		glm::vec3 worldPosDiff = newWorldPos - oldWorldPos;

		app->_mouseX = x;
		app->_mouseY = app->getHeight() - y;

		// Update GUI
		canvas->InputMouseMoved((int)x, (int)y, (int)dx, (int)dy);

		// Handle tools
		int leftButton = glfwGetMouseButton(app->_window, GLFW_MOUSE_BUTTON_LEFT);
		if (leftButton && app->_currentTool == Tools::SCROLL)
		{
			cam.translate(glm::vec3(-worldPosDiff.x, worldPosDiff.y, 0));
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
	}
	
	void VehicleSim::scroll_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Gwen::Controls::Canvas* canvas = app->_guiCanvas;

		// Update GUI
		canvas->InputMouseWheel((int)(y * 60.0));

		// Update app
		app->_orthoScale -= y;
		if (app->_orthoScale <= VEHICLESIM_MIN_SCALE)
			app->_orthoScale = VEHICLESIM_MIN_SCALE;
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
