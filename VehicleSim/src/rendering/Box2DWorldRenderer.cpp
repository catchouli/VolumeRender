#include "rendering/Box2DWorldRenderer.h"

#include <GL/glew.h>

#define _USE_MATH_DEFINES
#include <math.h>

namespace vlr
{
	Box2DWorldRenderer::Box2DWorldRenderer()
		: _world(nullptr)
	{
		initCircleMesh();
	}

	Box2DWorldRenderer::Box2DWorldRenderer(b2World* world)
		: _world(world)
	{
		initCircleMesh();
	}

	void Box2DWorldRenderer::render()
	{
		//static float diameter = 10.0f;
		////diameter += 0.01f;
		//fillCircle(2.5f, 0, diameter);

		// Render all bodies
		b2Body* bodies = _world->GetBodyList();

		for (b2Body* body = bodies; body; body = body->GetNext())
		{
			// Get all fixtures for this body
			b2Fixture* fixtures = body->GetFixtureList();

			for (b2Fixture* fixture = fixtures; fixture; fixture = fixture->GetNext())
			{
				switch (fixture->GetType())
				{
				case b2Shape::Type::e_polygon:
					{
						b2PolygonShape* shape = (b2PolygonShape*)fixture->GetShape();

						glMatrixMode(GL_MODELVIEW);
						glPushMatrix();

						// TODO: render;

						glPopMatrix();
					}
					break;
				case b2Shape::Type::e_circle:
					{
						b2CircleShape* shape = (b2CircleShape*)fixture->GetShape();

						glColor3f(1, 1, 1);
						fillCircle(body->GetPosition().x, body->GetPosition().y, shape->m_radius);
					}
					break;
				default:
					fprintf(stderr, "Fixture of invalid type found\n");
					break;
				}
			}
		}
	}

	void Box2DWorldRenderer::initCircleMesh()
	{
		// Calculate step
		float step = (float)((2 * M_PI) / (double)CIRCLE_DIVISIONS);
		
		// Build vertex array
		for (int i = 0; i < CIRCLE_DIVISIONS; ++i)
		{
			_circleVertexArray[2*i + 0] = 0.5f * cos(step * i);
			_circleVertexArray[2*i + 1] = 0.5f * sin(step * i);
		}
	}

	void Box2DWorldRenderer::fillCircle(float x, float y, float diameter)
	{
		// Switch to modelview matrix
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// Translate
		glTranslatef(x, y, 0);

		// Scale up
		glScalef(diameter, diameter, 1);

		// Render
		glColor3f(1, 1, 1);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(2, GL_FLOAT, 0, _circleVertexArray);
		glDrawArrays(GL_TRIANGLE_FAN, 0, CIRCLE_DIVISIONS);
		glDisableClientState(GL_VERTEX_ARRAY);

		// Restore modelview matrix
		glPopMatrix();
	}
}
