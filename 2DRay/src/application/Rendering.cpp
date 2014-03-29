#include "application/Ray2D.h"

#include "rendering/Raycast.h"
#include "maths/Types.h"
#include "util/Util.h"

using namespace vlr::rendering;

namespace vlr
{
	void Ray2D::render()
	{
		bool raycast = !glfwGetKey(_window, GLFW_KEY_G);

		// Clear screen
		glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Set up viewport
		glViewport(0, 0, getWidth(), getHeight());

		// Set up culling & depth testing
		glFrontFace(GL_CW);
		glEnable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		
		// Update opengl matrices
		_camera.updateGL();

		// Set up view
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Render raycasted
		int width = getWidth();
		int height = getHeight();

		if (raycast)
		{
			rendering::float4 origin;
			mat4 mvp;
			viewport viewport = _camera.getViewport();

			memcpy(origin.data, &_camera.getPos(), 3 * sizeof(float));
			origin.w = 1.0f;
			
			memcpy(mvp.data, &_camera.getMVP(), sizeof(mvp));

			renderOctree(_currentTree, &origin, &mvp, &viewport);
		}
		else
		{
			// Set up lighting
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);

			GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
			glLightfv(GL_LIGHT0, GL_POSITION, light_position);

			GLfloat diffConst[] = { 1.0f, 1.0f, 1.0f, 1.0 };
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffConst);

			// Render octree
			renderOctreeGL(_tree);
		
			glDisable (GL_LIGHTING);
			glLoadIdentity();
		}
	}
}
