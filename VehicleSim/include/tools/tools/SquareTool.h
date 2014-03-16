#ifndef VEHICLESIM_SQUARETOOL
#define VEHICLESIM_SQUARETOOL

#include "../Tool.h"

#include <math.h>

namespace vlr
{
	class SquareTool
		: public SelectionTool
	{
	public:
		SquareTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Rectangle tool")
		{
			float ypos = 0;

			_canDrag = false;
			_disableJointsWindow = true;

			_squareOptions = new Gwen::Controls::Base(_toolOptions);
			_squareOptions->Dock(Gwen::Pos::Fill);
			
			ypos = createBodyGui(_squareOptions, ypos);
			ypos = createFixtureGui(_squareOptions, ypos);

			_selectOne->SetText("");
		}

		virtual void update(float dt) override
		{
			if (_enabled)
			{
				SelectionTool::update(dt);

				if (_selected.size() > 0)
				{
					_squareOptions->Hide();
					_otherOptions->Show();
				}
				else
				{
					_squareOptions->Show();
					_otherOptions->Hide();
				}

				setText("Click and drag to draw a rectangle");
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

				float left = std::min(pos.x, startpos.x);
				float right = std::max(pos.x, startpos.x);
				float top = std::max(pos.y, startpos.y);
				float bottom = std::min(pos.y, startpos.y);
				b2Vec2 vertices[4] =
				{
					b2Vec2(left, top),
					b2Vec2(right, top),
					b2Vec2(right, bottom),
					b2Vec2(left, bottom)
				};

				_app->_worldRenderer.DrawSolidPolygon(vertices, 4, b2Color(1, 1, 1));
			}
		}

		virtual void click(int button, int action, int mods) override
		{
			if (_enabled && action == GLFW_RELEASE)
			{
				glm::vec2 startPos = worldSpace(_startX, _startY);
				glm::vec2 endPos = worldSpace(_x, _y);
					
				glm::vec2 minPos(std::min(startPos.x, endPos.x), std::min(startPos.y, endPos.y));
				glm::vec2 maxPos(std::max(startPos.x, endPos.x), std::max(startPos.y, endPos.y));
					
				glm::vec2 scale = maxPos - minPos;

				if (scale.x == 0 || scale.y == 0)
				{
					SelectionTool::click(button, action, mods);
					return;
				}

				glm::vec2 extents = scale * 0.5f;
				glm::vec2 midPoint = minPos + extents;

				// Create box at position
				b2BodyDef bodyDef = _bodyDef;

				bodyDef.position.Set(midPoint.x, midPoint.y);
				b2Body* body = _physWorld->CreateBody(&bodyDef);
					
				b2PolygonShape rect;
				rect.SetAsBox(extents.x, extents.y);

				b2FixtureDef fixtureDef = _fixtureDef;
				fixtureDef.shape = &rect;

				body->CreateFixture(&fixtureDef);

				// Reset tool
				SelectionTool::click(button, action, mods);
				_startX = _x;
				_startY = _y;

				// Set selection
				_selected.clear();
				_selected.push_back(body);
			}
		}

	protected:
		SquareTool(const SquareTool&);

		Gwen::Controls::Base* _squareOptions;
	};
}

#endif /* VEHICLESIM_SQUARETOOL */
