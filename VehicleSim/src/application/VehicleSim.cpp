#include "VehicleSim.h"

#include <util/Util.h>

// Input
#include "input/InputConverter.h"

namespace vlr
{
	const b2Vec2 gravity(0.0f, -50.0f);
	
	const float DEFAULT_SCALE = 10.0f;

	VehicleSim::VehicleSim()
		: Application(800, 600), _physWorld(gravity),
		_worldRenderer(&_physWorld), _lastPhysicsUpdate(glfwGetTime()),
		_orthoScale(DEFAULT_SCALE), _simulationRunning(false),
		_currentTool(nullptr), _currentToolButton(nullptr),
		_worldOptions(&_physWorld, &_worldRenderer),
		_timeStep(VEHICLESIM_PHYSICS_STEP),
		_velocityIterations(VEHICLESIM_VELOCITY_ITERATIONS),
		_positionIterations(VEHICLESIM_POSITION_ITERATIONS),
		_camFollow(nullptr)
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

		// Initialise box2d
		_physWorld.SetDebugDraw(&_worldRenderer);
		_worldRenderer.SetFlags(b2Draw::e_shapeBit | b2Draw::e_aabbBit | b2Draw::e_jointBit);

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
}
