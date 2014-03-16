#ifndef VEHICLESIM_CIRCLETOOL
#define VEHICLESIM_CIRCLETOOL

#include "../Tool.h"
#include "SelectionTool.h"

namespace vlr
{
	class CircleTool
		: public SelectionTool
	{
	public:
		CircleTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Circle Tool"), _radius(-1)
		{
			float ypos = 0;

			_canDrag = false;
			_disableJointsWindow = true;

			_circleOptions = new Gwen::Controls::Base(_toolOptions);
			_circleOptions->Dock(Gwen::Pos::Fill);
			
			ypos = createBodyGui(_circleOptions, ypos);
			ypos = createFixtureGui(_circleOptions, ypos);

			_selectOne->SetText("");
		}

		virtual void onEnabled() override
		{

		}

		virtual void onDisabled() override
		{

		}

		virtual void update(float dt) override
		{
			if (_enabled)
			{
				SelectionTool::update(dt);
				
				if (_selected.size() > 0)
				{
					_circleOptions->Hide();
					_otherOptions->Show();
				}
				else
				{
					_circleOptions->Show();
					_otherOptions->Hide();
				}

				setText("Click and drag to draw a circle");
			}			
		}

		virtual void render() override
		{
			if (_enabled)
				SelectionTool::render();

			if (_enabled && _dragging)
			{
				common::Camera& cam = _app->_camera;
				glm::vec2 pos = worldSpace(_x, _y);
				glm::vec2 startpos = worldSpace(_startX, _startY);

				_radius = glm::length(pos - startpos);

				_app->_worldRenderer.DrawSolidCircle(b2Vec2(startpos.x, startpos.y),
					_radius, b2Vec2(), b2Color(1, 1, 1));
			}
		}

		virtual void click(int button, int action, int mods) override
		{
			if (_enabled && action == GLFW_RELEASE)
			{
				if (_radius > 0)
				{
					glm::vec2 pos = worldSpace(_startX, _startY);

					// Create circle at position
					b2BodyDef bodyDef = _bodyDef;

					bodyDef.position.Set(pos.x, pos.y); 
					b2Body* body = _physWorld->CreateBody(&bodyDef);
					
					b2CircleShape circle;
					circle.m_p.Set(0, 0);
					circle.m_radius = _radius;

					b2FixtureDef fixtureDef = _fixtureDef; 
					fixtureDef.shape = &circle;

					body->CreateFixture(&fixtureDef);

					// Reset tool
					_radius = 0;
					SelectionTool::click(button, action, mods);
					_startX = _x;
					_startY = _y;

					// Set selection
					_selected.clear();
					_selected.push_back(body);
				}
				else
				{
					_moving = false;
					SelectionTool::click(button, action, mods);
				}
			}
		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{
			if (_radius > 0)
				_selected.clear();

			SelectionTool::mousemove(x, y, dx, dy);
		}

	protected:
		float _radius;

		Gwen::Controls::Base* _circleOptions;

		CircleTool(const CircleTool&);
	};
}

#endif /* VEHICLESIM_CIRCLETOOL */
