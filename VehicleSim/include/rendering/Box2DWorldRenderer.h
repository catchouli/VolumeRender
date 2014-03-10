#ifndef VEHICLESIM_BOX2DWORLDRENDERER
#define VEHICLESIM_BOX2DWORLDRENDERER

#include <Box2D/Box2D.h>

namespace vlr
{
	const int CIRCLE_DIVISIONS = 64;

	class Box2DWorldRenderer
	{
	public:
		Box2DWorldRenderer();
		Box2DWorldRenderer(b2World* world);

		void render();

	protected:
		Box2DWorldRenderer(const b2World&);

		void initCircleMesh();

		void fillCircle(float x, float y, float r);

	private:
		float _circleVertexArray[CIRCLE_DIVISIONS * 2];

		b2World* _world;
	};
}

#endif /* VEHICLESIM_BOX2DWORLDRENDERER */
