#ifndef VEHICLESIM_MOVEMENTTOOL
#define VEHICLESIM_MOVEMENTTOOL

#include "../Tool.h"

namespace vlr
{
	class MovementTool
		: public Tool
	{
	public:
		MovementTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: Tool(application, toolPanel, optionsPanel, icon, "Scrolling tool")
		{
			createNoOptions();
		}

		virtual void update(float dt) override
		{
			const float TRANSLATE_SPEED_SCREENPOS = 200.0f;

			float worldTranslateSpeed = glm::length(worldSpace(0, 0) -
				worldSpace(TRANSLATE_SPEED_SCREENPOS, 0));

			bool leftDown = glfwGetKey(_app->_window, GLFW_KEY_LEFT) > 0;
			bool rightDown = glfwGetKey(_app->_window, GLFW_KEY_RIGHT) > 0;
			bool upDown = glfwGetKey(_app->_window, GLFW_KEY_UP) > 0;
			bool downDown = glfwGetKey(_app->_window, GLFW_KEY_DOWN) > 0;

			glm::vec2 translate;

			if (leftDown || rightDown)
				translate.x += worldTranslateSpeed * dt * (leftDown ? -1 : 1);
			if (upDown || downDown)
				translate.y += worldTranslateSpeed * dt * (downDown ? -1 : 1);

			_camera->translate(glm::vec3(translate, 0));
		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{
			if (_enabled && _mousedown)
			{
				glm::vec2 oldCoords = worldSpace(_oldX, _oldY);
				glm::vec2 newCoords = worldSpace(_x, _y);

				glm::vec2 diff = oldCoords - newCoords;

				_camera->translate(glm::vec3(diff, 0));
			}
		}

	protected:
		MovementTool(const MovementTool&);
	};
}

#endif /* VEHICLESIM_MOVEMENTTOOL */
