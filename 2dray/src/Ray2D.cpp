#include "Ray2D.h"

#include <SDL2/SDL_opengl.h>

namespace vlr
{
	Ray2D::Ray2D()
		: common::SDLApplication(800, 600)
	{

	}

	void Ray2D::initGL()
	{
		// Set clear colour
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		// Set up viewport
		glViewport(0, 0, getWidth(), getHeight());

		// Set up orthographic projection
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0f, getWidth(), getHeight(), 0.0f, 0.0f, 1.0f);

		// Initialise modelview matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	void Ray2D::initialise()
	{
		// Initialise opengl
		initGL();
	}

	void Ray2D::update(float dt)
	{
		// Update window title with FPS
		const int TITLE_BUFFER_LEN = 1024;
		const char* TITLE_FORMAT = "FPS: %d";
		char titleBuffer[TITLE_BUFFER_LEN];

		snprintf(titleBuffer, TITLE_BUFFER_LEN, TITLE_FORMAT, getFPS());

		setTitle(titleBuffer);
	}

	void Ray2D::render()
	{
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void Ray2D::handleEvent(SDL_Event event)
	{

	}
}
