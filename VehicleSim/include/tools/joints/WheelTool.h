#ifndef VEHICLESIM_WHEELTOOL
#define VEHICLESIM_WHEELTOOL

#include "../Tool.h"
#include "../tools/SelectionTool.h"

namespace vlr
{
	class WheelTool
		: public SelectionTool
	{
	protected:
		enum Mode;

	public:
		WheelTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Wheel joint tool"),
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
			case MODE_GROUNDANCHOR1:
			case MODE_SELECT2:
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
			case MODE_GROUNDANCHOR1:
				setText("Select direction from anchor for axis of constraint (hint: above the anchor?)");
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
				_anchor1world = b2Vec2(worldPos.x, worldPos.y);
				_mode = MODE_GROUNDANCHOR1;
				break;
			case MODE_ANCHOR2:
				_anchor2world = b2Vec2(worldPos.x, worldPos.y);
				createJoint();
				break;
			case MODE_GROUNDANCHOR1:
				_groundAnchor1 = b2Vec2(worldPos.x, worldPos.y);
				_mode = MODE_SELECT2;
				break;
			}
		}

	protected:
		WheelTool(const WheelTool&);

		void createJoint()
		{
			b2WheelJointDef jointDef;

			jointDef.collideConnected = false;

			jointDef.bodyA = _selected[0];
			jointDef.bodyB = _selected[1];
			jointDef.localAnchorA = jointDef.bodyA->GetLocalPoint(_anchor1world);
			jointDef.localAnchorB = jointDef.bodyB->GetLocalPoint(_anchor2world);

			b2Vec2 axis = _groundAnchor1 - _anchor1world;
			axis.Normalize();
			jointDef.localAxisA = axis;

			b2WheelJoint* joint = (b2WheelJoint*)_physWorld->CreateJoint(&jointDef);
			joint->SetUserData(new MotorInput(_motorInput));

			reset();
		}

		b2Vec2 _anchor1world, _anchor2world;
		b2Vec2 _groundAnchor1;
		b2WheelJointDef _wheelJointDef;

		enum Mode
		{
			MODE_SELECT1,
			MODE_ANCHOR1,
			MODE_GROUNDANCHOR1,
			MODE_SELECT2,
			MODE_ANCHOR2
		};

		Mode _mode;
	};
}

#endif /* VEHICLESIM_WHEELTOOL */
