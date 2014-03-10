#ifndef VEHICLESIM_APPLICATION
#define VEHICLESIM_APPLICATION

// Application headers
#include "app/Application.h"
#include "app/Framebuffer.h"
#include "rendering/Camera.h"
#include "rendering/Box2DWorldRenderer.h"

// Box2D
#include <Box2D/Box2D.h>

// GWEN headers
#include <Gwen/Gwen.h>
#include <Gwen/Renderers/OpenGL_DebugFont.h>

// Gwen controls
#include <Gwen/Skins/TexturedBase.h>
#include <Gwen/Controls/DockBase.h>

// Standard headers
#include <stdio.h>
#include <vector>

#include "Resources.h"

namespace vlr
{
	const int VEHICLESIM_PHYSICS_STEP = 60;
	const float VEHICLESIM_PHYSICS_STEP_TIME = 1.0f / (float)VEHICLESIM_PHYSICS_STEP;

	const float VEHICLESIM_MIN_SCALE = 2.0f;

	namespace Tools
	{
		enum Tool
		{
			NONE,
			SCROLL,
			CIRCLE
		};
	}

	typedef Tools::Tool Tool;

	class VehicleSim
		: public Gwen::Event::Handler, public common::Application
	{
	public:
		enum Mode;

		VehicleSim();
		~VehicleSim();

		void update(double dt);

		void render();

		void setMode(Mode newMode);

		// Input Callbacks
		static void mouse_move_callback(GLFWwindow* window,
			double x, double y);
		static void mouse_callback(GLFWwindow* window, int button,
			int action, int mods);
		static void key_callback(GLFWwindow* window, int key,
			int scancode, int action, int mods);
		static void scroll_callback(GLFWwindow* window, double x,
			double y);
		static void char_callback(GLFWwindow* window,
			unsigned int codepoint);
		static void resize_callback(GLFWwindow* window,
			int width, int height);

		// GUI callbacks
		void setSimulationRunning(Gwen::Event::Info);
		void selectTool(Gwen::Event::Info info);
		void deselectTool(Gwen::Event::Info info);

		enum Mode
		{
			MODE_SIMULATE,
			MODE_LEVELEDIT,
			MODE_OBJECTEDIT
		};

	protected:
		VehicleSim(const VehicleSim&);

		void resize(int width, int height);

		void initGui();

	private:
		// State
		Tool _currentTool;
		Gwen::Controls::Base* _currentToolButton;

		bool _simulationRunning;
		Mode _mode;
		double _lastPhysicsUpdate;
		std::vector<glm::vec2> _points;

		// Camera
		common::Camera _camera;
		float _width, _height, _aspect;
		float _orthoScale;

		double _mouseX, _mouseY;

		// Box2d
		b2World _physWorld;

		// Box2d renderer
		Box2DWorldRenderer _worldRenderer;

		// Gwen
		Gwen::Renderer::OpenGL* _guiRenderer;
		Gwen::Skin::TexturedBase* _guiSkin;
		Gwen::Controls::Canvas* _guiCanvas;
		Gwen::Controls::DockBase* _guiDock;

		// GUI elements
		Gwen::Controls::Button* _simButton;
		std::vector<Gwen::Controls::Button*> _toolButtons;
	};

}

#endif /* VEHICLESIM_APPLICATION */
