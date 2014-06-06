#include "application/VolumeRender.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <glm/glm.hpp>

#include "rendering/Raycast.h"
#include "rendering/Rendering.h"
#include "maths/Types.h"
#include "util/Util.h"

using namespace vlr::rendering;

namespace vlr
{
	void VolumeRender::render()
	{
		bool raycast = !glfwGetKey(_window, GLFW_KEY_G);

		// Update camera
		_camera.updateGL();

		// Set up viewport
		glViewport(0, 0, getWidth(), getHeight());

		// Add the ability to change the light position and direction with the 1 key
		if (glfwGetKey(_window, GLFW_KEY_1))
		{
			vlr::rendering::light_t* dir_light = &rendering_attributes.lights[0];
			dir_light->position = rendering_attributes.origin;
			dir_light->direction = glm::vec3(0, 0, -1.0f) * _camera.getRot();
		}

		// Set up origin, mvp, viewport
		rendering_attributes.origin = _camera.getPos();
		rendering_attributes.mvp = _camera.getMVP();
		rendering_attributes.viewport = _camera.getViewport();

		// Clear screen
		glClearColor(rendering_attributes.clear_colour.r,
					 rendering_attributes.clear_colour.g,
					 rendering_attributes.clear_colour.b,
					 rendering_attributes.clear_colour.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		// Render octree
		renderOctree(_gpuTree, rendering_attributes);

		// Render mesh with opengl
		if (rendering_attributes.settings.enable_depth_copy)
		{
			// Set up OpenGL state
			// Set up culling & depth testing
			glFrontFace(GL_CW);
			glEnable(GL_CULL_FACE);
		
			// Set up view
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
		
			//Add ambient light
			GLfloat ambientColor[] = {0.2f, 0.2f, 0.2f, 1.0f}; //Color(0.2, 0.2, 0.2)
			glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientColor);
		
			//Add positioned light
			GLfloat lightColor0[] = {0.1f, 0.1f, 0.1f, 1.0f}; //Color (0.5, 0.5, 0.5)
			GLfloat lightPos0[] = {4.0f, 0.0f, 8.0f, 1.0f}; //Positioned at (4, 0, 8)
			glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
			glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

			// Render mesh
			glPushMatrix();
			glScalef(0.1f, 0.1f, 0.1f);
			_mesh.render();
			glPopMatrix();
		}

		// Check for opengl errors
		checkGlError();
	}
}
