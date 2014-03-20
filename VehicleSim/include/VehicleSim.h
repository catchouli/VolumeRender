#ifndef VEHICLESIM_APPLICATION
#define VEHICLESIM_APPLICATION

// Application headers
#include "app/Application.h"
#include "app/Framebuffer.h"
#include "rendering/Camera.h"
#include "rendering/Box2DWorldRenderer.h"
#include "tools/Tool.h"

// Box2D
#include <Box2D/Box2D.h>

// GWEN headers
#include <Gwen/Gwen.h>
#include <Gwen/Renderers/OpenGL_DebugFont.h>

// Gwen controls
#include <Gwen/Skins/TexturedBase.h>
#include <Gwen/Controls/DockBase.h>
#include <Gwen/Controls/TabButton.h>

// Options
#include "tools/gui/FloatOption.h"
#include "options/WorldOptions.h"

// Standard headers
#include <stdio.h>
#include <vector>
#include <unordered_map>

#include "Resources.h"

namespace vlr
{
	const int VEHICLESIM_PHYSICS_STEP = 60;
	const int VEHICLESIM_VELOCITY_ITERATIONS = 6;
	const int VEHICLESIM_POSITION_ITERATIONS = 2;

	const float VEHICLESIM_MIN_SCALE = 2.0f;

	class VehicleSim
		: public Gwen::Event::Handler, public common::Application
	{
	public:
		VehicleSim();
		~VehicleSim();

		void doStep();
		void update(double dt);

		void render();

		void doFrameInput(float dt);

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

		void disableSim();

		// GUI callbacks
		void setSimulationRunning(Gwen::Event::Info);
		void resetSimulation(Gwen::Event::Info);
		void selectTool(Gwen::Event::Info info);
		void deselectTool(Gwen::Event::Info info);

		// Menu bar
		void getString(Gwen::Event::Info info);
		void newDocument(Gwen::Event::Info info);
		void saveDocument(Gwen::Event::Info info);
		void saveDocumentAs(Gwen::Event::Info info);
		void loadDocument(Gwen::Event::Info info);
		void importDocument(Gwen::Event::Info info);
		void exitApplication(Gwen::Event::Info info);

	protected:
		VehicleSim(const VehicleSim&);

		void resize(int width, int height);

		void initGui();

	private:
		friend class Serialiser;

		friend class Tool;
		friend class MovementTool;
		friend class CircleTool;
		friend class SquareTool;
		friend class PolyTool;
		friend class ZoomTool;
		friend class SelectionTool;
		friend class RotateTool;
		friend class CamFollow;
		
		friend class NoCollideTool;
		friend class DistanceTool;
		friend class RevoluteTool;
		friend class PrismaticTool;
		friend class PulleyTool;
		friend class WheelTool;
		friend class WeldTool;

		// State
		int _timeStep;
		int _velocityIterations;
		int _positionIterations;

		std::string _storedState;
		Tool* _currentTool;
		std::vector<Tool*> _tools;
		Gwen::Controls::Base* _currentToolButton;

		bool _gotString;
		std::string _lastString;
		std::string _filename;

		bool _simulationRunning;
		double _lastPhysicsUpdate;
		std::vector<glm::vec2> _points;
		std::unordered_map<std::string, Gwen::Controls::Base*> _options;

		// Camera
		common::Camera _camera;
		b2Body* _camFollow;
		CamFollow* _cf;
		float _width, _height, _aspect;
		float _orthoScale;

		double _mouseX, _mouseY;

		// Box2d
		b2World _physWorld;

		// Box2d renderer
		Box2DWorldRenderer _worldRenderer;

		//std::vector<

		// Gwen
		Gwen::Renderer::OpenGL* _guiRenderer;
		Gwen::Skin::TexturedBase* _guiSkin;
		Gwen::Controls::Canvas* _guiCanvas;
		Gwen::Controls::DockBase* _guiDock;
		Gwen::Controls::Label* _statusLabel;

		WorldOptions _worldOptions;
		std::vector<Updatable*> _updatableOptions;
		Gwen::Controls::TabButton* _worldOptionsTabButton;

		// GUI elements
		Gwen::Controls::Button* _simButton;
		Gwen::Controls::Button* _resetButton;
		std::vector<Gwen::Controls::Button*> _toolButtons;
	};

}

#endif /* VEHICLESIM_APPLICATION */
