#ifndef VEHICLESIM_POLYTOOL
#define VEHICLESIM_POLYTOOL

#include "../Tool.h"

#include "poly2tri/poly2tri.h"

#include <Gwen/Controls/CheckBox.h>
#include <math.h>

double dot(float ax, float ay, float bx, float by);
double perpDot(float ax, float ay, float bx, float by);
bool lineCollision(float ax, float ay, float bx, float by,
                    float cx, float cy, float dx, float dy);

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
					_firstX = pos.x;
					_firstY = pos.y;
					_vertices.push_back(b2Vec2(pos.x, pos.y));
				}
				else
				{
					// Check if this is an attempt to close the polygon
					glm::vec2 screenStartPos = screenSpace(_firstX, _firstY);
					float diff = glm::length(glm::vec2(_x, _y) - screenStartPos);

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

						// Triangulate polygon
						std::vector<p2t::Point*> points;
						for (unsigned int i = 0; i < _vertices.size(); ++i)
						{
							points.push_back(new p2t::Point(_vertices[i].x, _vertices[i].y));
						}

						p2t::CDT cdt(points);
						cdt.Triangulate();
						std::vector<p2t::Triangle*> tris = cdt.GetTriangles();

						for (unsigned int i = 0; i < tris.size(); ++i)
						{
							std::vector<b2Vec2> triPoints;
							
							triPoints.push_back(b2Vec2(tris[i]->GetPoint(0)->x, tris[i]->GetPoint(0)->y));
							triPoints.push_back(b2Vec2(tris[i]->GetPoint(1)->x, tris[i]->GetPoint(1)->y));
							triPoints.push_back(b2Vec2(tris[i]->GetPoint(2)->x, tris[i]->GetPoint(2)->y));

							b2PolygonShape poly;
							poly.Set(triPoints.data(), triPoints.size());

							b2FixtureDef fixtureDef = _fixtureDef;
							fixtureDef.shape = &poly;

							body->CreateFixture(&fixtureDef);
						}

						// Reset tool
						reset();
						
						_selected.clear();
						_selected.push_back(body);

						return;
					}

					bool intersectsAny = false;
					b2Vec2 last = _vertices.back();

					for (unsigned int i = 1; i < _vertices.size(); ++i)
					{
						if (lineCollision(_vertices[i-1].x, _vertices[i-1].y,
							_vertices[i].x, _vertices[i].y,
							pos.x, pos.y,
							last.x, last.y))
						{
							intersectsAny = true;
							break;
						}
					}

					if (!intersectsAny)
						_vertices.push_back(b2Vec2(pos.x, pos.y));
				}
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
