#ifndef VEHICLESIM_NOCOLLIDETOOL
#define VEHICLESIM_NOCOLLIDETOOL

#include "../Tool.h"
#include "../tools/SelectionTool.h"

namespace vlr
{
	class NoCollideTool
		: public SelectionTool
	{
	protected:
		enum Mode;

	public:
		NoCollideTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "No collide tool"),
			_mode(MODE_SELECT1)
		{
			_multiselectRequiresShift = false; 
			_maxSelected = 2;
			_canDrag = false;

			_disableOptions = true;
			_disableJointsWindow = true;

			createNoOptions();
		}

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
				_mode = MODE_SELECT2;
				break;
			case MODE_SELECT2:
				createJoint();
				reset();
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
			case MODE_SELECT2:
				_mode = MODE_SELECT1;
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
			case MODE_SELECT2:
				setText("Select body 2");
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
			}
		}

	protected:
		NoCollideTool(const NoCollideTool&);

		void createJoint()
		{
			//b2DistanceJointDef jointDef = _distanceJointDef;
			//jointDef.collideConnected = false;
			//jointDef.bodyA = _selected[0];
			//jointDef.bodyB = _selected[1];
			//jointDef.localAnchorA = _anchor1;
			//jointDef.localAnchorB = _anchor2;
			//jointDef.length = (_anchor2w - _anchor1w).Length();
			//b2DistanceJoint* joint = (b2DistanceJoint*)_physWorld->CreateJoint(&jointDef);

			// Check body joints
			for (b2JointEdge* jointEdge = _selected[0]->GetJointList(); jointEdge; jointEdge = jointEdge->next)
			{
				b2Joint* joint = jointEdge->joint;

				// If this is a no collide joint
				if (joint->GetType() == b2JointType::e_frictionJoint)
				{
					// Check the bodies
					if ((_selected[0] == joint->GetBodyA() && _selected[1] == joint->GetBodyB()) ||
						(_selected[0] == joint->GetBodyB() && _selected[1] == joint->GetBodyA()))
					{
						// If this would create the same joint, return
						return;
					}
				}
			}

			b2FrictionJointDef jointDef;
			jointDef.collideConnected = false;
			jointDef.bodyA = _selected[0];
			jointDef.bodyB = _selected[1];
			jointDef.localAnchorA = jointDef.bodyA->GetLocalPoint(jointDef.bodyA->GetPosition());
			jointDef.localAnchorB = jointDef.bodyB->GetLocalPoint(jointDef.bodyB->GetPosition());
			
			b2Joint* joint = _physWorld->CreateJoint(&jointDef);

			reset();
		}
		
		b2Vec2 _anchor1, _anchor2;
		b2Vec2 _anchor1w, _anchor2w;
		b2DistanceJointDef _distanceJointDef;

		enum Mode
		{
			MODE_SELECT1,
			MODE_SELECT2
		};

		Mode _mode;
	};
}

#endif /* VEHICLESIM_NOCOLLIDETOOL */
