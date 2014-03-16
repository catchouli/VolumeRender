#ifndef VEHICLESIM_BOX2DWORLDRENDERER
#define VEHICLESIM_BOX2DWORLDRENDERER

#include <GL/glew.h>
#include <Box2D/Box2D.h>
#include <Box2D/Common/b2Draw.h>

namespace vlr
{
	const int CIRCLE_DIVISIONS = 64;

	class Box2DWorldRenderer
		: public b2Draw
	{
	public:
		Box2DWorldRenderer();
		Box2DWorldRenderer(b2World* world);

		void render();

		virtual void DrawPolygon(const b2Vec2* vertices, int32 vertexCount,
			const b2Color& color) override;
		virtual void DrawSolidPolygon(const b2Vec2* vertices, int32 vertexCount,
			const b2Color& color) override;
		virtual void DrawCircle(const b2Vec2& center, float32 radius,
			const b2Color& color) override;
		virtual void DrawSolidCircle(const b2Vec2& center, float32 radius,
			const b2Vec2& axis, const b2Color& color) override;
		virtual void DrawSegment(const b2Vec2& p1, const b2Vec2& p2,
			const b2Color& color) override;
		virtual void DrawTransform(const b2Transform& xf) override;
		
		void drawCircle(float x, float y, float r);
		void fillCircle(float x, float y, float r);

		void drawPoly(const b2Vec2* vertices, int32 vertexCount);

		void drawPolyClosed(const b2Vec2* vertices, int32 vertexCount);

		void drawPoints(const b2Vec2* vertices, int32 vertexCount);

	protected:
		Box2DWorldRenderer(const b2World&);

		void initCircleMesh();
		void renderCircle(float x, float y, float r, GLenum mode);

	private:
		float _circleVertexArray[CIRCLE_DIVISIONS * 2];

		b2World* _world;
	};
}

#endif /* VEHICLESIM_BOX2DWORLDRENDERER */
