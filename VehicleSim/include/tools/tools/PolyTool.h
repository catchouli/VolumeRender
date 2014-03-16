#ifndef VEHICLESIM_POLYTOOL
#define VEHICLESIM_POLYTOOL

#include "../Tool.h"

#include <Gwen/Controls/CheckBox.h>
#include <math.h>

namespace vlr
{
	const float POLYTOOL_CLOSE_DISTANCE = 5.0f;

	class PolyTool
		: public SelectionTool
	{
	public:
		PolyTool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel, Gwen::Controls::Base* optionsPanel, const char* icon)
			: SelectionTool(application, toolPanel, optionsPanel, icon, "Polygon tool")
		{
			float ypos = 0;

			_canDrag = false;
			_disableJointsWindow = true;

			_polyOptions = new Gwen::Controls::Base(_toolOptions);
			_polyOptions->Dock(Gwen::Pos::Fill);
			
			ypos = createBodyGui(_polyOptions, ypos);
			ypos = createFixtureGui(_polyOptions, ypos);

			_selectOne->SetText("");
		}

		virtual void reset() override
		{
			_vertices.clear();
			_firstX = _firstY = 0;

			SelectionTool::reset();
		}

		virtual void render() override
		{
			if (_enabled)
			{
				glPointSize(5);

				glColor3f(1, 1, 1);
				_app->_worldRenderer.drawPoints(_vertices.data(), _vertices.size());
				_app->_worldRenderer.drawPoly(_vertices.data(), _vertices.size());

				glPointSize(1);
			}

			SelectionTool::render();
		}

		virtual void update(float dt) override
		{
			if (_vertices.size() == 0 && _selected.size() > 0)
			{
				_polyOptions->Hide();
				_otherOptions->Show();
			}
			else
			{
				_polyOptions->Show();
				_otherOptions->Hide();
			}

			SelectionTool::update(dt);
		}

		virtual void click(int button, int action, int mods) override
		{
			if (_enabled && action == GLFW_PRESS)
			{
				glm::vec2 pos = worldSpace(_x, _y);

				// Set initial X
				if (_vertices.size() == 0)
				{
					_firstX = _x;
					_firstY = _y;
				}
				else
				{
					// Check if this is an attempt to close the polygon
					float diff = glm::length(glm::vec2(_x, _y) - glm::vec2(_firstX, _firstY));

					if (diff <= POLYTOOL_CLOSE_DISTANCE && _vertices.size() > 2)
					{
						// Average the points to get a suitable centre
						b2Vec2 centre(0, 0);

						for (unsigned int i = 0; i < _vertices.size(); ++i)
						{
							centre += _vertices[i];
						}
						
						centre.x /= (float)_vertices.size();
						centre.y /= (float)_vertices.size();

						// Subtract centre from the points to turn them into offsets
						for (unsigned int i = 0; i < _vertices.size(); ++i)
						{
							_vertices[i] -= centre;
						}

						// Close polygon
						b2BodyDef bodyDef = _bodyDef;

						bodyDef.position.Set(centre.x, centre.y);
						b2Body* body = _physWorld->CreateBody(&bodyDef);
					
						b2PolygonShape poly;
						poly.Set(_vertices.data(), _vertices.size());

						b2FixtureDef fixtureDef = _fixtureDef;
						fixtureDef.shape = &poly;

						body->CreateFixture(&fixtureDef);

						// Reset tool
						reset();
						
						_selected.clear();
						_selected.push_back(body);

						return;
					}
				}

				_vertices.push_back(b2Vec2(pos.x, pos.y));
			}

			if (_vertices.size() > 0)
				_selected.clear();
		}

		virtual void mousemove(double x, double y, double dx, double dy) override
		{
			SelectionTool::mousemove(x, y, dx, dy);
		}

	protected:
		PolyTool(const PolyTool&);

		Gwen::Controls::Base* _polyOptions;

		float _firstX, _firstY;

		std::vector<b2Vec2> _vertices;
	};
}

#endif /* VEHICLESIM_POLYTOOL */
