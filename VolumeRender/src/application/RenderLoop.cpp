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

		// Rotate directional
		glm::vec3 lightDir = glm::vec3(0.0f, 0.0f, -1.0f) * glm::rotate(glm::quat(), (float)glfwGetTime() * 10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
		lightDir = glm::vec3(0, 0, -1);

		// Set up origin, mvp, viewport
		rendering_attributes.origin = _camera.getPos();
		rendering_attributes.mvp = _camera.getMVP();
		rendering_attributes.viewport = _camera.getViewport();

		// Set up rendering attributes for scene
		// Lights
		rendering_attributes.light_count = 1;

		rendering_attributes.ambient_colour = glm::vec3(0.1f, 0.1f, 0.1f);

		light_t* dir_light = &rendering_attributes.lights[0];
		
		dir_light->type = rendering::LightTypes::DIRECTIONAL;
		dir_light->diffuse = glm::vec3(0.3f, 0.3f, 0.3f);
		dir_light->specular = glm::vec3(0.5f, 0.5f, 0.5f);
		dir_light->direction = lightDir;
		
		dir_light->constant_att = 1.0f;
		dir_light->linear_att = 0.0f;
		dir_light->quadratic_att = 0.0f;

		for (int i = 1; i < 10; ++i)
		{
			dir_light = &rendering_attributes.lights[i];
		
			dir_light->type = rendering::LightTypes::DIRECTIONAL;
			dir_light->diffuse = glm::vec3(0, 0, 0.3f);
			dir_light->specular = glm::vec3(0.5f, 0.5f, 0.5f) * 0.0f;
			dir_light->direction = lightDir;
		
			dir_light->constant_att = 1.0f;
			dir_light->linear_att = 0.0f;
			dir_light->quadratic_att = 0.0f;
		}

		// Render octree
		renderOctree(_gpuTree, rendering_attributes);

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
