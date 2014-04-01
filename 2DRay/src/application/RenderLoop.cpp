#include "application/Ray2D.h"

#include "rendering/Rendering.h"
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

		// Prepare to render octree
		rendering::float4 origin;
		mat4 mvp;
		viewport viewport;
		
		// Update viewport, origin and mvp
		viewport = _camera.getViewport();

		memcpy(origin.data, &_camera.getPos(), 3 * sizeof(float));
		origin.w = 1.0f;
			
		memcpy(mvp.data, &_camera.getMVP(), sizeof(mvp));

		// Render octree
		renderOctree(_gpuTree, &origin, &mvp, &viewport);
	}
}
