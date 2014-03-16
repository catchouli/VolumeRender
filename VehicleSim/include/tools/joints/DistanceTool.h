#ifndef VEHICLESIM_DISTANCETOOL
#define VEHICLESIM_DISTANCETOOL

#include "../Tool.h"
#include "../tools/SelectionTool.h"

namespace vlr
{
	class DistanceTool
		: public SelectionTool
	{
	protected:
		enum Mode;

	public:
		DistanceTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Distance joint tool"),
			_mode(MODE_SELECT1)
		{
			_multiselectRequiresShift = false; 
			_maxSelected = 2;
			_canDrag = false;

			_disableOptions = true;
			_disableJointsWindow = true;

			initGui();
		}

		void initGui();

		virtual void reset() override
		{
			_mode = MODE_SELECT1;

			SelectionTool::reset();
		}

		virtual void render() override
		{
			SelectionTool::render();
		}

		virtual void onSelect(b2Body*) override
		{
			switch (_mode)
			{
			case MODE_SELECT1:
				_mode = MODE_ANCHOR1;
				break;
			case MODE_SELECT2:
				_mode = MODE_ANCHOR2;
				break;
			default:
				fprintf(stderr, "Object selected when none expected\n");
				break;
			}
		}

		virtual void onDeselect(b2Body*) override
		{
			switch (_mode)
			{
			case MODE_ANCHOR1:
				_mode = MODE_SELECT1;
				break;
			case MODE_ANCHOR2:
				_mode = MODE_SELECT2;
				break;
			}
		}

		virtual void update(float) override
		{
			if (!_enabled)
				return;

			switch (_mode)
			{
			case MODE_SELECT1:
				setText("Select body 1");
				break;
			case MODE_ANCHOR1:
				setText("Select anchor point for body 1");
				break;
			case MODE_SELECT2:
				setText("Select body 2");
				break;
			case MODE_ANCHOR2:
				setText("Select anchor point for body 2");
				break;
			default:
				break;
			}
		}

		virtual void click(int button, int action, int mods) override
		{
			if (!_enabled)
				return;

			if (action != GLFW_RELEASE)
				return;

			glm::vec2 worldPos = worldSpace(_x, _y);

			switch (_mode)
			{
			case MODE_SELECT1:
				SelectionTool::click(button, action, mods);
				break;
			case MODE_SELECT2:
				SelectionTool::click(button, action, mods);
				break;
			case MODE_ANCHOR1:
				_anchor1w = b2Vec2(worldPos.x, worldPos.y);
				_anchor1 = _selected[0]->GetLocalPoint(b2Vec2(worldPos.x, worldPos.y));
				_mode = MODE_SELECT2;
				break;
			case MODE_ANCHOR2:
				_anchor2w = b2Vec2(worldPos.x, worldPos.y);
				_anchor2 = _selected[1]->GetLocalPoint(b2Vec2(worldPos.x, worldPos.y));

				createJoint();

				break;
			}
		}

	protected:
		DistanceTool(const DistanceTool&);

		void createJoint()
		{
			b2DistanceJointDef jointDef;
			jointDef.collideConnected = false;
			jointDef.bodyA = _selected[0];
			jointDef.bodyB = _selected[1];
			jointDef.localAnchorA = _anchor1;
			jointDef.localAnchorB = _anchor2;
			jointDef.length = (_anchor2w - _anchor1w).Length();
			b2DistanceJoint* joint = (b2DistanceJoint*)_physWorld->CreateJoint(&jointDef);

			reset();
		}
		
		b2Vec2 _anchor1, _anchor2;
		b2Vec2 _anchor1w, _anchor2w;
		b2DistanceJointDef _distanceJointDef;

		enum Mode
		{
			MODE_SELECT1,
			MODE_ANCHOR1,
			MODE_SELECT2,
			MODE_ANCHOR2
		};

		Mode _mode;
	};
}

#endif /* VEHICLESIM_DISTANCETOOL */
