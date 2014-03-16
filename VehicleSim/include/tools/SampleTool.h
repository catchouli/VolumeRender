#ifndef VEHICLESIM_SAMPLETOOL
#define VEHICLESIM_SAMPLETOOL

#include "Tool.h"

namespace vlr
{
	class SampleTool
		: public Tool
	{
	public:
		SampleTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: Tool(application, toolPanel, optionsPanel, icon)
		{

		}

		virtual void update(float dt) override
		{

		}

		virtual void click(int button, int action, int mods) override
		{

		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{

		}

		virtual void key(int key, int scancode, int action, int mods) override
		{

		}

		virtual void scroll(double x, double y) override
		{

		}

	protected:
		SampleTool(const SampleTool&);
	};
}

#endif /* VEHICLESIM_SAMPLETOOL */
