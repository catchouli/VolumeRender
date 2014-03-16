#ifndef VEHICLESIM_PULLEYTOOL
#define VEHICLESIM_PULLEYTOOL

#include "../Tool.h"
#include "../tools/SelectionTool.h"

namespace vlr
{
	class PulleyTool
		: public SelectionTool
	{
	protected:
		enum Mode;

	public:
		PulleyTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Pulley joint tool"),
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
			case MODE_GROUNDANCHOR2:
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
				setText("Select ground anchor for body 1");
				break;
			case MODE_SELECT2:
				setText("Select body 2");
				break;
			case MODE_ANCHOR2:
				setText("Select anchor point for body 2");
				break;
			case MODE_GROUNDANCHOR2:
				setText("Select ground anchor for body 2");
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
				_mode = MODE_GROUNDANCHOR2;
				break;
			case MODE_GROUNDANCHOR1:
				_groundAnchor1 = b2Vec2(worldPos.x, worldPos.y);
				_mode = MODE_SELECT2;
				break;
			case MODE_GROUNDANCHOR2:
				_groundAnchor2 = b2Vec2(worldPos.x, worldPos.y);
				createJoint();
				break;
			}
		}

	protected:
		PulleyTool(const PulleyTool&);

		void createJoint()
		{
			b2PulleyJointDef jointDef;

			jointDef.collideConnected = false;

			jointDef.bodyA = _selected[0];
			jointDef.bodyB = _selected[1];
			jointDef.localAnchorA = jointDef.bodyA->GetLocalPoint(_anchor1world);
			jointDef.localAnchorB = jointDef.bodyB->GetLocalPoint(_anchor2world);
			jointDef.groundAnchorA = _groundAnchor1;
			jointDef.groundAnchorB = _groundAnchor2;
			jointDef.lengthA = (_anchor1world - _groundAnchor1).Length();
			jointDef.lengthB = (_anchor2world - _groundAnchor2).Length();

			b2PulleyJoint* joint = (b2PulleyJoint*)_physWorld->CreateJoint(&jointDef);

			reset();
		}

		b2Vec2 _anchor1world, _anchor2world;
		b2Vec2 _groundAnchor1, _groundAnchor2;
		b2PulleyJointDef _pulleyJointDef;

		enum Mode
		{
			MODE_SELECT1,
			MODE_ANCHOR1,
			MODE_GROUNDANCHOR1,
			MODE_SELECT2,
			MODE_ANCHOR2,
			MODE_GROUNDANCHOR2
		};

		Mode _mode;
	};
}

#endif /* VEHICLESIM_PULLEYTOOL */
