#ifndef VEHICLESIM_CAMFOLLOW
#define VEHICLESIM_CAMFOLLOW

#include "../Tool.h"
#include "../tools/SelectionTool.h"

namespace vlr
{
	class CamFollow
		: public SelectionTool
	{
	protected:
		enum Mode;

	public:
		CamFollow(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Camera follow tool")
		{
			_multiselectRequiresShift = false; 
			_maxSelected = 1;
			_canDrag = false;

			_disableOptions = true;
			_disableJointsWindow = true;

			createNoOptions();
		}

		virtual void update(float) override
		{
			if (!_enabled)
				return;

			setText("Select a body to follow");

			if (_selected.size() == 1)
			{
				_app->_camFollow = _selected[0];
				reset();
			}

			if (_app->_camFollow != nullptr)
			{
				_camera->setPos(glm::vec3(_app->_camFollow->GetPosition().x, _app->_camFollow->GetPosition().y, 0));
			}
		}

		virtual void click(int button, int action, int mods) override
		{
			if (button == GLFW_MOUSE_BUTTON_LEFT)
			{
				SelectionTool::click(button, action, mods);
			}
			else if (button == GLFW_MOUSE_BUTTON_RIGHT)
			{
				_selected.clear();
				_app->_camFollow = nullptr;
			}
		}

	protected:
		CamFollow(const CamFollow&);
	};
}

#endif /* VEHICLESIM_CAMFOLLOW */
