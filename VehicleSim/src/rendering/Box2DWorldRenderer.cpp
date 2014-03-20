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

	void Box2DWorldRenderer::DrawPolygon(const b2Vec2* vertices, int32 vertexCount,
		const b2Color& color)
	{

		drawPolyClosed(vertices, vertexCount);
	}

	void Box2DWorldRenderer::DrawSolidPolygon(const b2Vec2* vertices, int32 vertexCount,
		const b2Color& color)
	{
		// Enable blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		// Set colour
		glColor4f(color.r, color.b, color.b, 0.5f);

		// Enable vertex arrays
		glEnableClientState(GL_VERTEX_ARRAY);

		// Set vertex pointer
		glVertexPointer(2, GL_FLOAT, 0, vertices);

		// Draw polygon
		glDrawArrays(GL_POLYGON, 0, vertexCount);

		// Disable vertex arrays
		glDisableClientState(GL_VERTEX_ARRAY);

		// Draw outline
		DrawPolygon(vertices, vertexCount, color);

		// Disable blending
		glDisable(GL_BLEND);
	}

	void Box2DWorldRenderer::DrawCircle(const b2Vec2& center, float32 radius,
		const b2Color& color)
	{
		// Draw circle
		glColor3f(color.r, color.g, color.b);
		drawCircle(center.x, center.y, radius*2);
		glPointSize(3);
		glBegin(GL_POINTS);
		glVertex2f(center.x, center.y);
		glEnd();
		glPointSize(1);
	}

	void Box2DWorldRenderer::DrawSolidCircle(const b2Vec2& center, float32 radius,
		const b2Vec2& axis, const b2Color& color)
	{
		// Enable blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		// Fill circle
		glColor4f(color.r, color.g, color.b, 0.5f);
		fillCircle(center.x, center.y, radius*2);

		// Draw outline
		glColor4f(color.r, color.g, color.b, 1.0f);
		DrawCircle(center, radius, color);

		glColor3f(1, 1, 1);
		glBegin(GL_LINES);
		glVertex2f(center.x, center.y);
		glVertex2f(center.x + axis.x * radius, center.y + axis.y * radius);
		glEnd();

		// Disable blending
		glDisable(GL_BLEND);
	}

	void Box2DWorldRenderer::DrawSegment(const b2Vec2& p1, const b2Vec2& p2,
		const b2Color& color)
	{
		glColor3f(color.r, color.g, color.b);

		glBegin(GL_LINES);
		glVertex2f(p1.x, p1.y);
		glVertex2f(p2.x, p2.y);
		glEnd();
	}

	void Box2DWorldRenderer::DrawTransform(const b2Transform& xf)
	{

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
	
	void Box2DWorldRenderer::drawCircle(float x, float y, float diameter)
	{
		renderCircle(x, y, diameter, GL_LINE_STRIP);
	}

	void Box2DWorldRenderer::fillCircle(float x, float y, float diameter)
	{
		renderCircle(x, y, diameter, GL_TRIANGLE_FAN);
	}

	void Box2DWorldRenderer::drawPoly(const b2Vec2* vertices, int32 vertexCount)
	{
		// Enable vertex arrays
		glEnableClientState(GL_VERTEX_ARRAY);

		// Set vertex pointer
		glVertexPointer(2, GL_FLOAT, 0, vertices);

		// Draw polygon
		glDrawArrays(GL_LINE_STRIP, 0, vertexCount);

		// Disable vertex arrays
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	void Box2DWorldRenderer::drawPolyClosed(const b2Vec2* vertices, int32 vertexCount)
	{
		// Enable vertex arrays
		glEnableClientState(GL_VERTEX_ARRAY);

		// Set vertex pointer
		glVertexPointer(2, GL_FLOAT, 0, vertices);

		// Draw polygon
		glDrawArrays(GL_LINE_LOOP, 0, vertexCount);

		// Disable vertex arrays
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	void Box2DWorldRenderer::drawPoints(const b2Vec2* vertices, int32 vertexCount)
	{
		// Enable vertex arrays
		glEnableClientState(GL_VERTEX_ARRAY);

		// Set vertex pointer
		glVertexPointer(2, GL_FLOAT, 0, vertices);

		// Draw polygon
		glDrawArrays(GL_POINTS, 0, vertexCount);

		// Disable vertex arrays
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	void Box2DWorldRenderer::renderCircle(float x, float y, float diameter, GLenum mode)
	{
		// Switch to modelview matrix
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// Translate
		glTranslatef(x, y, 0);

		// Scale up
		glScalef(diameter, diameter, 1);

		// Render
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(2, GL_FLOAT, 0, _circleVertexArray);
		glDrawArrays(mode, 0, CIRCLE_DIVISIONS);
		glDisableClientState(GL_VERTEX_ARRAY);

		// Restore modelview matrix
		glPopMatrix();
	}
}
