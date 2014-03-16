#include "VehicleSim.h"

namespace vlr
{
	void VehicleSim::render()
	{
		// Clear background
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Reset opengl state at the start of each frame
		// Since gwen's opengl renderer is too rude to do it itself
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_BLEND);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

		// Update from camera
		_camera.updateGL();

		// Render
		_physWorld.DrawDebugData();

		// Render tools
		std::vector<Tool*>& tools = _tools;
		for (auto it = tools.begin(); it != tools.end(); ++it)
		{
			(*it)->render();
		}

		// Reset matrices
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		float width = (float)_width;
		float height = (float)_height;
		glOrtho(0, width, height, 0, -1.0f, 1.0f);
		glViewport(0, 0, width, height);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Render GUI
		_guiCanvas->RenderCanvas();
	}
}
