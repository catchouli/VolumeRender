#include "application/VolumeRender.h"

#include <glm/glm.hpp>

#include "rendering/Rendering.h"
#include "maths/Types.h"
#include "util/Util.h"

using namespace vlr::rendering;

namespace vlr
{
	void VolumeRender::render()
	{
		bool raycast = !glfwGetKey(_window, GLFW_KEY_G);

		// Clear screen
		glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Update camera
		_camera.updateGL();

		// Set up viewport
		glViewport(0, 0, getWidth(), getHeight());

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

		//// Render mesh with opengl
		//// Set up OpenGL state
		//// Set up culling & depth testing
		//glFrontFace(GL_CW);
		//glEnable(GL_CULL_FACE);
		//glEnable(GL_DEPTH_TEST);
		//
		//// Set up view
		//glMatrixMode(GL_MODELVIEW);
		//glLoadIdentity();

		//glEnable(GL_LIGHTING);
		//glEnable(GL_LIGHT0);
		//glEnable(GL_DEPTH_TEST);
		//
		////Add ambient light
		//GLfloat ambientColor[] = {0.2f, 0.2f, 0.2f, 1.0f}; //Color(0.2, 0.2, 0.2)
		//glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientColor);
		//
		////Add positioned light
		//GLfloat lightColor0[] = {0.1f, 0.1f, 0.1f, 1.0f}; //Color (0.5, 0.5, 0.5)
		//GLfloat lightPos0[] = {4.0f, 0.0f, 8.0f, 1.0f}; //Positioned at (4, 0, 8)
		//glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
		//glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

		//// Render mesh
		//glPushMatrix();
		//glScalef(0.1f, 0.1f, 0.1f);
		//_mesh.render();
		//glPopMatrix();
	}
}
