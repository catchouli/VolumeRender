#ifndef VEHICLESIM_SELECTIONTOOL
#define VEHICLESIM_SELECTIONTOOL

#include "../Tool.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <Gwen/Controls/ListBox.h>

#include "VehicleSim.h"
#include "input/InputConverter.h"
#include "rendering/Box2DWorldRenderer.h"
#include "tools/gui/OptionBase.h"

namespace vlr
{
	class SelectionTool
		: public Gwen::Event::Handler, public Tool
	{
	public:
		SelectionTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel,
			Gwen::Controls::Base* optionsPanel, const char* icon, const char* name = "Selection tool")
			: Tool(application, toolPanel, optionsPanel, icon, name),
			_moving(false), _multiselectRequiresShift(true), _maxSelected(-1),
			_canDrag(true), _name(name), _disableOptions(false),
			_disableJointsWindow(false)
		{
			// Create joints page
			if (_jointsPage == nullptr)
			{
				_jointsPage = new Gwen::Controls::Base(getDock());
				_jointsPage->Dock(Gwen::Pos::Fill);

				// Initialise joints page
				_listBox = new Gwen::Controls::ListBox(_jointsPage);
				_listBox->Dock(Gwen::Pos::Fill);
				_listBox->onRowSelected.Add(this, &SelectionTool::jointSelectCallback);

				// Create joints options page
				_jointOptionsPanel = new Gwen::Controls::Base(_jointsPage);
				_jointOptionsPanel->SetWidth(_jointsPage->GetSize().x);
				_jointOptionsPanel->SetHeight(390);
				_jointOptionsPanel->Dock(Gwen::Pos::Bottom);

				// Add joints page to dock
				_jointsButton = 
					((Gwen::Controls::TabControl*)optionsPanel)->
					AddPage("Attached Joints", _jointsPage);
				_jointsButton->Hide();
			}

			_selectOne = new Gwen::Controls::Label(_toolOptions);
			_selectOne->SetText("Select one object");
			_selectOne->Dock(Gwen::Pos::Fill);

			_otherOptions = new Gwen::Controls::Base(_toolOptions);
			_otherOptions->Dock(Gwen::Pos::Fill);

			initGui();
		}

		void initGui();
		void selectJoint(b2Joint* joint);

		float createJointInputGuiSelection(
		Gwen::Controls::Base* parent, float ypos, b2Joint* joint);

		void jointSelectCallback(Gwen::Event::Info info)
		{
			b2Joint* joint = ((Gwen::Controls::ListBox*)info.Control)->
				GetSelectedRow()->UserData.Get<b2Joint*>("joint");

			if (joint != _currentJoint)
			{
				for (auto it = _jointOptionsUpdatables.begin(); it != _jointOptionsUpdatables.end(); ++it)
				{
					(*it)->setEnabled(false);
				}
				_jointOptionsUpdatables.clear();

				_currentJoint = joint;
				selectJoint(_currentJoint);
			}
		}

		void deleteJoint(Gwen::Event::Info info)
		{
			if (_currentJoint != nullptr)
			{
				_physWorld->DestroyJoint(_currentJoint);
				_currentJoint = nullptr;

				_listBox->Clear();
				getJoints(_currentBody);
				_jointOptions->Hide();

				for (auto it = _jointOptionsUpdatables.begin(); it != _jointOptionsUpdatables.end(); ++it)
				{
					(*it)->setEnabled(false);
				}
				_jointOptionsUpdatables.clear();
			}
		}

		virtual void onSelect(b2Body* body)
		{
			// Update joints page
			delete _jointOptions;
			_jointOptions = nullptr;
			_currentJoint = nullptr;
			
			// clear joint options
			// TODO:
			// Clean up (can't work out how - causes an exception in Gwen)
			// or this causes a memory leak
			for (auto it = _jointOptionsUpdatables.begin(); it != _jointOptionsUpdatables.end(); ++it)
			{
				(*it)->setEnabled(false);
			}
			_jointOptionsUpdatables.clear();

			_listBox->Clear();
			if (_selected.size() == 1)
			{
				_currentBody = body;
				for (auto it = _bodyOptions.begin(); it != _bodyOptions.end(); ++it)
				{
					(*it)->setBase(body);
				}
				getJoints(body);
			}
		}

		virtual void onDeselect(b2Body* body)
		{
			// Update joints page
			delete _jointOptions;
			_jointOptions = nullptr;
			_currentJoint = nullptr;

			for (auto it = _bodyOptions.begin(); it != _bodyOptions.end(); ++it)
			{
				(*it)->setBase(nullptr);
			}
			
			// clear joint options
			// TODO:
			// Clean up (can't work out how - causes an exception in Gwen)
			// or this causes a memory leak
			for (auto it = _jointOptionsUpdatables.begin(); it != _jointOptionsUpdatables.end(); ++it)
			{
				(*it)->setEnabled(false);
			}

			_jointOptionsUpdatables.clear();
			_inputButtons.clear();

			_listBox->Clear();
		}

		void getJoints(b2Body* body)
		{
			Gwen::Controls::Layout::TableRow* last;

			int i = 0;
			for (b2JointEdge* jointEdge = body->GetJointList();
				jointEdge; jointEdge = jointEdge->next)
			{
				b2Joint* joint = jointEdge->joint;

				std::string typeStr;

				switch (joint->GetType())
				{
				case b2JointType::e_distanceJoint:
					typeStr = "Distance Joint";
					break;
				case b2JointType::e_frictionJoint:
					typeStr = "No collide";
					break;
				case b2JointType::e_motorJoint:
					typeStr = "Motor Joint";
					break;
				case b2JointType::e_mouseJoint:
					typeStr = "Mouse Joint";
					break;
				case b2JointType::e_prismaticJoint:
					typeStr = "Prismatic Joint";
					break;
				case b2JointType::e_pulleyJoint:
					typeStr = "Pulley Joint";
					break;
				case b2JointType::e_revoluteJoint:
					typeStr = "Revolute Joint";
					break;
				case b2JointType::e_ropeJoint:
					typeStr = "Rope Joint";
					break;
				case b2JointType::e_weldJoint:
					typeStr = "Weld Joint";
					break;
				case b2JointType::e_wheelJoint:
					typeStr = "Wheel Joint";
					break;
				case b2JointType::e_unknownJoint:
					typeStr = "Unknown Joint";
					break;
				}

				auto item = _listBox->AddItem(typeStr);
				item->UserData.Set("joint", joint);

				i++;

				last = item;
			}
		}

		virtual void reset() override
		{
			std::vector<b2Body*> removedBodies(_selected);
			_selected.clear();
			
			// Clear selected list
			for (auto it = removedBodies.begin(); it != removedBodies.end(); ++it)
			{
				onDeselect(*it);
			}
		}

		virtual void render() override
		{
			if (_disableOptions)
			{
				_selectOne->Hide();
				_otherOptions->Hide();
			}

			// Render outlines on selected bodies
			for (unsigned int i = 0; i < _selected.size(); ++i)
			{
				b2Body* body = _selected[i];

				// Render all fixtures
				for (b2Fixture* fixture = body->GetFixtureList(); fixture; fixture = fixture->GetNext())
				{
					b2Shape* shape = fixture->GetShape();

					if (shape != nullptr)
					{
						switch (shape->GetType())
						{
						case b2Shape::e_circle:
							{
								b2CircleShape* circle = (b2CircleShape*)shape;

								float radius = circle->m_radius;
								b2Vec2 pos = circle->m_p + body->GetPosition();

								// Render circle
								glColor3f(1, 1, 0);
								glLineWidth(3);
								_renderer->drawCircle(pos.x, pos.y, radius*2);
								glLineWidth(1);
							}
							break;
						case b2Shape::e_polygon:
							{
								b2PolygonShape* poly = (b2PolygonShape*)shape;

								const b2Vec2& vertices = poly->GetVertex(0);

								// Render poly
								// Transform to position
								glColor3f(1, 1, 0);
								glMatrixMode(GL_MODELVIEW);
								glPushMatrix();
								glTranslatef(body->GetPosition().x, body->GetPosition().y, 0);
								glRotatef(body->GetAngle() * (180.0f / M_PI), 0, 0, 1);

								// Make line thicker
								glLineWidth(3);
								
								_renderer->drawPolyClosed(&vertices, poly->GetVertexCount());

								// Restore opengl state
								glPopMatrix();
								glLineWidth(1);
							}
							break;
						}
					}
				}
			}
		}

		virtual void update(float dt) override
		{
			if (!_enabled || _disableOptions)
				return;
			
			// Update joint options
			for (auto it = _jointOptionsUpdatables.begin();
				it != _jointOptionsUpdatables.end(); ++it)
			{
				(*it)->update();
			}

			// Show options if one body selected
			if (_selected.size() == 1)
			{
				if (!_disableJointsWindow)
					_jointsButton->Show();
				_otherOptions->Show();
				_selectOne->Hide();

				for (auto it = _bodyOptions.begin(); it != _bodyOptions.end(); ++it)
				{
					(*it)->setBase(_selected[0]);
				}

				// Get fixture count
				b2Fixture* firstFixture = _selected[0]->GetFixtureList();

				if (firstFixture != nullptr)
				{
					for (auto it = _fixtureOptions.begin(); it != _fixtureOptions.end(); ++it)
					{
						(*it)->setBase(firstFixture);
					}

					// Duplicate settings of all fixtures
					for (b2Fixture* fixture = firstFixture->GetNext(); fixture; fixture = fixture->GetNext())
					{
						fixture->SetFriction(firstFixture->GetFriction());
						fixture->SetRestitution(firstFixture->GetRestitution());
						fixture->SetDensity(firstFixture->GetDensity());
					}
				}
			}
			else
			{
				_jointsButton->Hide();
				_selectOne->Show();
				_otherOptions->Hide();

				for (auto it = _bodyOptions.begin(); it != _bodyOptions.end(); ++it)
				{
					(*it)->setBase(nullptr);
				}

				for (auto it = _fixtureOptions.begin(); it != _fixtureOptions.end(); ++it)
				{
					(*it)->setBase(nullptr);
				}
			}
		}

		virtual void click(int button, int action, int mods) override
		{
			bool bodyFound = false;

			if (!_enabled)
				return;

			if (action == GLFW_RELEASE)
			{
				if (_moving)
				{
					_moving = false;

					// Enable bodies
					for (unsigned int i = 0; i < _selected.size(); ++i)
					{
						_selected[i]->SetAwake(true);
					}

					return;
				}

				// Create AABB at mouse point
				glm::vec2 worldPoint = worldSpace(_x, _y);

				// Iterate through every body
				for (b2Body* body = _physWorld->GetBodyList(); body; body = body->GetNext())
				{
					bool alreadySelected = std::find(_selected.begin(), _selected.end(), body) != _selected.end();

					bool shiftHeld = (mods & GLFW_MOD_SHIFT) > 0;
					bool altHeld = (mods & GLFW_MOD_ALT) > 0;

					if (altHeld && !alreadySelected)
						continue;
					if (!altHeld && alreadySelected)
						continue;

					// Iterate through every fixture
					for (b2Fixture* fixture = body->GetFixtureList(); fixture; fixture = fixture->GetNext())
					{
						bool active = body->IsActive();

						// Make body active for a moment if it's not
						// (inactive bodies don't have AABBs for some reason)
						body->SetActive(true);
						const b2AABB& aabb = fixture->GetAABB(0);
						body->SetActive(active);

						if (aabb.lowerBound.x <= worldPoint.x &&
							aabb.lowerBound.y <= worldPoint.y &&
							aabb.upperBound.x >= worldPoint.x &&
							aabb.upperBound.y >= worldPoint.y)
						{
							if (!altHeld && _maxSelected >= 0 && (int)_selected.size() >= _maxSelected)
								return;

 							bodyFound = true;

							// If alt held (remove mode)
							if (altHeld)
							{
								_selected.erase(std::remove(_selected.begin(), _selected.end(), body), _selected.end());

								onDeselect(body);

								goto selectiontool_endloop;
							}
							else
							{
								// Clear selection list if shift not held
								if (_multiselectRequiresShift && !shiftHeld)
									reset();
								
								_selected.push_back(body);
								onSelect(body);

								goto selectiontool_endloop;
							}
						}
					}
				}

	selectiontool_endloop:
				// Deselect if no body found
				if (!bodyFound)
					reset();
			}
		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{
			const float MIN_DRAG = 5.0f;

			if (_enabled && _mousedown)
			{
				if (!_moving)
				{
					if (glm::length(glm::vec2(_startX, _startY) - glm::vec2(_x, _y)) > MIN_DRAG)
						_moving = true;
				}

				if (_moving && _canDrag)
				{
					// Calculate mouse diff this frame in world space
					glm::vec2 worldCur = worldSpace(_x, _y);
					glm::vec2 worldOld = worldSpace(_oldX, _oldY);

					// Calculate difference
					glm::vec2 diff = worldCur - worldOld;

					// Translate all selected bodies
					for (unsigned int i = 0; i < _selected.size(); ++i)
					{
						b2Vec2 pos = _selected[i]->GetPosition();
						float angle = _selected[i]->GetAngle();

						pos += b2Vec2(diff.x, diff.y);

						_selected[i]->SetTransform(pos, angle);
						_selected[i]->SetAwake(false);
					}
				}
			}
		}

		virtual void key(int key, int scancode, int action, int mods) override
		{
			if (action == GLFW_RELEASE)
			{
				// Update input buttons
				for (auto it = _inputButtons.begin();
					it != _inputButtons.end(); ++it)
				{
					if ((*it)->GetToggleState())
					{
						*(*it)->UserData.Get<int*>("updateKey") = key;
						(*it)->SetText(InputConverter::translateCharToString(key));
						(*it)->SetToggleState(false);
					}
				}

				if (key == GLFW_KEY_DELETE)
				{
					for (unsigned int i = 0; i < _selected.size(); ++i)
					{
						b2Body* body = _selected[i];

						if (_app->_camFollow == body)
							_app->_camFollow = nullptr;

						_physWorld->DestroyBody(body);
					}

					reset();
				}

				if ((mods & GLFW_MOD_CONTROL) > 0 && key == GLFW_KEY_D)
				{
					// Dulpicate each body
					for (auto it = _selected.begin(); it != _selected.end(); ++it)
					{
						b2Body* body = *it;

						b2BodyDef bodyDef;
						bodyDef.active = body->IsActive();
						bodyDef.type = body->GetType();
						bodyDef.position = body->GetPosition();
						bodyDef.angle = body->GetAngle();
						bodyDef.linearDamping = body->GetLinearDamping();
						bodyDef.linearVelocity = body->GetLinearVelocity();
						bodyDef.angularDamping = body->GetAngularDamping();
						bodyDef.angularVelocity = body->GetAngularVelocity();
						bodyDef.allowSleep = body->IsSleepingAllowed();
						bodyDef.awake = body->IsAwake();
						bodyDef.bullet = body->IsBullet();
						bodyDef.fixedRotation = body->IsFixedRotation();
						bodyDef.active = body->IsActive();
						bodyDef.gravityScale = body->GetGravityScale();

						b2Body* newBody = _physWorld->CreateBody(&bodyDef);

						for (b2Fixture* fix = body->GetFixtureList(); fix; fix = fix->GetNext())
						{
							b2FixtureDef fixDef;
							fixDef.density = fix->GetDensity();
							fixDef.filter = fix->GetFilterData();
							fixDef.friction = fix->GetFriction();
							fixDef.isSensor = fix->IsSensor();
							fixDef.restitution = fix->GetRestitution();

							b2Shape* shape = fix->GetShape();
							
							switch (shape->GetType())
							{
							case b2Shape::e_circle:
								{
									b2CircleShape circle;
									circle.m_p = ((b2CircleShape*)shape)->m_p;
									circle.m_radius = ((b2CircleShape*)shape)->m_radius;

									fixDef.shape = &circle;

									newBody->CreateFixture(&fixDef);
								}
								break;
							case b2Shape::e_polygon:
								{
									b2PolygonShape poly;
									poly.Set(&((b2PolygonShape*)shape)->GetVertex(0), ((b2PolygonShape*)shape)->GetVertexCount());

									fixDef.shape = &poly;

									newBody->CreateFixture(&fixDef);
								}
								break;
							default:
								break;
							}
						}
					}
				}
			}
		}

		virtual void scroll(double x, double y) override
		{

		}

	protected:
		SelectionTool(const SelectionTool&);
		
		std::vector<OptionBase<b2Body>*> _bodyOptions;
		std::vector<OptionBase<b2Fixture>*> _fixtureOptions;

		std::vector<Updatable*> _jointOptionsUpdatables;

		std::vector<Gwen::Controls::Button*> _inputButtons;

		bool _canDrag;
		bool _multiselectRequiresShift;
		int _maxSelected;
		bool _moving;
		std::string _name;
		std::vector<b2Body*> _selected;

		bool _disableOptions;
		bool _disableJointsWindow;
		
		Gwen::Controls::Label* _selectOne;
		Gwen::Controls::Base* _otherOptions;

		static Gwen::Controls::Base* _jointsPage;
		static Gwen::Controls::Base* _jointOptions;
		static Gwen::Controls::Base* _jointOptionsPanel;
		static Gwen::Controls::ListBox* _listBox;
		static Gwen::Controls::TabButton* _jointsButton;

		static b2Body* _currentBody;
		static b2Joint* _currentJoint;
	};
}

#endif /* VEHICLESIM_SELECTIONTOOL */
