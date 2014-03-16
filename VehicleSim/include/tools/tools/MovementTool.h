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
