#ifndef VEHICLESIM_ZOOMTOOL
#define VEHICLESIM_ZOOMTOOL

#include "../Tool.h"

#include "VehicleSim.h"

namespace vlr
{
	const float MIN_ORTHO_SCALE = 1.0f;
	const float MAX_ORTHO_SCALE = 100.0f;

	const float MOUSEMOVE_ZOOM_SPEED = 0.1f;

	class ZoomTool
		: public Tool
	{
	public:
		ZoomTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: Tool(application, toolPanel, optionsPanel, icon, "Zooming tool")
		{
			createNoOptions();
		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{
			if (_enabled && _mousedown)
			{
				// Sum diffs
				float diff = dx + (-dy);
				float amount = diff * MOUSEMOVE_ZOOM_SPEED;
				_app->_orthoScale = b2Clamp(_app->_orthoScale - amount,
					MIN_ORTHO_SCALE, MAX_ORTHO_SCALE);
			}
		}

		virtual void scroll(double x, double y) override
		{
			_app->_orthoScale = b2Clamp(_app->_orthoScale - (float)y,
				MIN_ORTHO_SCALE, MAX_ORTHO_SCALE);
		}

	protected:
		ZoomTool(const ZoomTool&);
	};
}

#endif /* VEHICLESIM_ZOOMTOOL */
