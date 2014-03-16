#ifndef VEHICLESIM_ROTATETOOL
#define VEHICLESIM_ROTATETOOL

#include "../Tool.h"
#include "SelectionTool.h"

namespace vlr
{
	class RotateTool
		: public SelectionTool
	{
	public:
		RotateTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Rotation tool")
		{
			_multiselectRequiresShift = false; 
			_maxSelected = 1;
			_canDrag = false;
		}

		virtual void reset() override
		{
			SelectionTool::reset();
		}

		virtual void render() override
		{
			SelectionTool::render();
		}

		virtual void click(int button, int action, int mods) override
		{
			if (action != GLFW_RELEASE)
				return;

			if (_selected.size() == 2)
			{
				glm::vec2 worldPos = worldSpace(_x, _y);

				b2RevoluteJointDef jointDef;
				jointDef.bodyA = _selected[0];
				jointDef.bodyB = _selected[1];
				jointDef.localAnchorA = _selected[0]->GetLocalPoint(b2Vec2(worldPos.x, worldPos.y));
				b2RevoluteJoint* joint = (b2RevoluteJoint*)_physWorld->CreateJoint(&jointDef);

				_selected.clear();
			}
			else
			{
				SelectionTool::click(button, action, mods);
			}
		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{
			SelectionTool::mousemove(x, y, dx, dy);

			// Rotate selected body whilst dragging
			if (_enabled && _mousedown && _dragging && (int)_selected.size() > 0)
			{
				b2Body* body = _selected[0];

				body->SetTransform(body->GetPosition(), body->GetAngle() + dx / 180.0f);
			}
		}

	protected:
		RotateTool(const RotateTool&);
	};
}

#endif /* VEHICLESIM_ROTATETOOL */
