#ifndef VEHICLESIM_TOOL
#define VEHICLESIM_TOOL

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <Gwen/Controls/Button.h>
#include <Gwen/Controls/DockBase.h>
#include <Gwen/Controls/Label.h>
#include <Gwen/Controls/TabControl.h>
#include <Gwen/Controls/Layout/Tile.h>
#include <Box2D/Box2D.h>
#include <rendering/Camera.h>

#include "input/MotorInput.h"
#include "misc/Updatable.h"

#include <unordered_map>

namespace vlr
{
	class VehicleSim;
	class Box2DWorldRenderer;

	class Tool
	{
	public:
		Tool(VehicleSim* application, Gwen::Controls::Layout::Tile* toolPanel,
			Gwen::Controls::Base* optionsPanel, const char* icon,
			const char* name = "Unconfigured tool");

		// Utils
		glm::vec2 worldSpace(float x, float y);
		Gwen::Controls::DockBase* getDock();

		// State
		void setEnabled(bool value);

		void setText(const char* text);

		virtual void onEnabled();
		virtual void onDisabled();

		virtual void reset();
		void resetOptions();

		void update_base(float dt);
		virtual void update(float dt);

		virtual void render();

		// Input
		void click_base(int button, int action, int mods);
		virtual void click(int button, int action, int mods);

		void mousemove_base(double x, double y, double dx, double dy);
		virtual void mousemove(double x, double y, double dx, double dy);

		void key_base(int key, int scancode, int action, int mods);
		virtual void key(int key, int scancode, int action, int mods);

		void scroll_base(double x, double y);
		virtual void scroll(double x, double y);

		void showOptions();
		void hideOptions();

		void createNoOptions();
		float createBodyGui(Gwen::Controls::Base* parent, float ypos);
		float createFixtureGui(Gwen::Controls::Base* parent, float ypos);
		float createJointInputGui(Gwen::Controls::Base* parent, float ypos);

	protected:
		std::vector<Updatable*> _updatableOptions;

		static b2BodyDef _bodyDef;
		static b2FixtureDef _fixtureDef;

		bool _enabled;

		VehicleSim* _app;
		b2World* _physWorld;
		Box2DWorldRenderer* _renderer;
		common::Camera* _camera;
		std::unordered_map<std::string, Gwen::Controls::Base*>* _options;

		static MotorInput _motorInput;
		static std::vector<Gwen::Controls::Button*> _forwardButtons;
		static std::vector<Gwen::Controls::Button*> _reverseButtons;

		Gwen::Controls::Button* _button;
		Gwen::Controls::TabButton* _tabButton;
		Gwen::Controls::Base* _toolOptions;
		Gwen::Controls::Base* _optionsPanel;
		Gwen::Controls::Label* _statusLabel;

		bool _dragging;
		bool _mousedown;
		float _startX, _startY;
		float _x, _y;
		float _oldX, _oldY;
	};
}

#endif /* VEHICLESIM_TOOL */
