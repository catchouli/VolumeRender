#include "VehicleSim.h"

#include <util/Util.h>

#include "input/InputConverter.h"

namespace vlr
{
	const char* fontDir = "assets/fonts";

	const b2Vec2 gravity(0.0f, -10.0f);
	
	const float DEFAULT_SCALE = 10.0f;

	VehicleSim::VehicleSim()
		: Application(800, 600), _mode(MODE_SIMULATE), _physWorld(gravity),
		_worldRenderer(&_physWorld), _lastPhysicsUpdate(glfwGetTime()),
		_orthoScale(DEFAULT_SCALE), _simulationRunning(false),
		_currentTool(Tools::NONE), _currentToolButton(nullptr)
	{
		// Update window pointer
		glfwSetWindowUserPointer(_window, this);

		// Set callbacks
 		glfwSetCursorPosCallback(_window, mouse_move_callback);
		glfwSetMouseButtonCallback(_window, mouse_callback);
		glfwSetKeyCallback(_window, key_callback);
		glfwSetScrollCallback(_window, scroll_callback);
		glfwSetWindowSizeCallback(_window, resize_callback);
		glfwSetCharCallback(_window, char_callback);

		// Initialise gwen
		_guiRenderer = new Gwen::Renderer::OpenGL_DebugFont();
		_guiRenderer->Init();

		_guiSkin = new Gwen::Skin::TexturedBase(_guiRenderer);
		_guiSkin->Init("DefaultSkin.png");

		_guiCanvas = new Gwen::Controls::Canvas(_guiSkin);
		_guiCanvas->SetSize(getWidth(), getHeight());
		
		// Initialise camera & viewport etc
		resize(getWidth(), getHeight());

		// Initialise input state
		glfwGetCursorPos(_window, &_mouseX, &_mouseY);

		// Initialise GUI
		initGui();
	}

	VehicleSim::~VehicleSim()
	{

	}

	void VehicleSim::update(double dt)
	{
		const float MOVE_SPEED = 2.0f;

		// Get inner area of dock
		Gwen::Rect innerBounds = _guiDock->GetInnerBounds();

		// Update camera's viewport
		_camera.setViewport(innerBounds.x, innerBounds.y, innerBounds.w, innerBounds.h);

		// Update camera's matrix
		if (innerBounds.h > 0)
		{
			float aspect = (float)innerBounds.w / innerBounds.h;
			_camera.orthographic(_orthoScale, aspect);
		}

		// Update physics system
		double time = glfwGetTime();
		while (_lastPhysicsUpdate + VEHICLESIM_PHYSICS_STEP_TIME < time)
		{
			if (_simulationRunning)
				_physWorld.Step(VEHICLESIM_PHYSICS_STEP_TIME, 6, 2);

			_lastPhysicsUpdate += VEHICLESIM_PHYSICS_STEP_TIME;
		}

		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, "FPS: %d\n", getFPS());
		glfwSetWindowTitle(_window, title);
	}

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
		_worldRenderer.render();

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

	void VehicleSim::setMode(Mode newMode)
	{
		_mode = newMode;
	}
}
