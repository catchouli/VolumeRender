#ifndef VEHICLESIM_TOOL
#define VEHICLESIM_TOOL

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <Gwen/Controls/Base.h>
#include <Gwen/Controls/Button.h>
#include <Gwen/Controls/DockBase.h>
#include <Gwen/Controls/Label.h>
#include <Gwen/Controls/TabControl.h>
#include <Gwen/Controls/Layout/Tile.h>
#include <Box2D/Box2D.h>
#include <rendering/Camera.h>

#include "input/MotorInput.h"
#include "misc/Updatable.h"
#include "tools/gui/FloatOption.h"
#include "tools/gui/IntOption.h"
#include "tools/gui/BoolOption.h"
#include "tools/gui/MultiOption.h"
#include "tools/gui/VectorOption.h"
#include "tools/gui/SliderOption.h"

#include <unordered_map>

namespace vlr
{
	const float LINE_HEIGHT = 22;
	const float OPTIONS_X_START = 120;
	const float OPTIONS_X_WIDTH = 220;

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
		glm::vec2 screenSpace(float x, float y);

		static Gwen::Controls::Label* createLabel(const char* string,
			Gwen::Controls::Base* base, float x, float y);

		template <typename baseType>
		inline static BoolOption<baseType>* createBoolOption(const char* string,
			Gwen::Controls::Base* base,
			float x, float y, GetterSetter<baseType, bool> getter);

		template <typename baseType>
		inline static FloatOption<baseType>* createFloatOption(const char* string,
			Gwen::Controls::Base* base,
			float x, float y, GetterSetter<baseType, float> getter);

		template <typename baseType>
		inline static SliderOption<baseType>* createSliderOption(const char* string,
			Gwen::Controls::Base* base, float x, float y,
			GetterSetter<baseType, float> getter, float min, float max);

		template <typename baseType>
		inline static VectorOption<baseType>* createVectorOption(const char* string,
			Gwen::Controls::Base* base,
			float x, float y, GetterSetter<baseType, b2Vec2> getter);

		template <typename baseType>
		inline static IntOption<baseType>* createIntOption(const char* string,
			Gwen::Controls::Base* base,
			float x, float y, GetterSetter<baseType, int> getter);

		template <typename baseType, typename optionType>
		inline static MultiOption<baseType, optionType>*
			createMultiOption(const char* string,
			Gwen::Controls::Base* base, float x, float y,
			GetterSetter<baseType, optionType> getter);

		Gwen::Controls::DockBase* getDock();

		// State
		void setEnabled(bool value);

		void setText(const char* text);

		virtual void onEnabled();
		virtual void onDisabled();

		virtual void reset();

		void setSelected(bool val);
		bool getSelected() const;

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

	// Boolean options
	template <typename baseType>
	BoolOption<baseType>* Tool::createBoolOption(const char* string,
		Gwen::Controls::Base* base,
		float x, float y, GetterSetter<baseType, bool> getter)
	{
		// Create label
		createLabel(string, base, x, y);

		// Create checkbox
		BoolOption<baseType>* bo = 
			new BoolOption<baseType>(base, getter);
		bo->getCheckBox()->SetPos(OPTIONS_X_START, y);

		return bo;
	}

	// Float options
	template <typename baseType>
	FloatOption<baseType>* Tool::createFloatOption(const char* string,
		Gwen::Controls::Base* base,
		float x, float y, GetterSetter<baseType, float> getter)
	{
		// Create label
		createLabel(string, base, x, y);

		// Create textbox
		FloatOption<baseType>* fo = 
			new FloatOption<baseType>(base, getter);
		fo->getTextBox()->SetPos(OPTIONS_X_START, y);
		fo->getTextBox()->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);

		return fo;
	}

	// Float options
	template <typename baseType>
	SliderOption<baseType>* Tool::createSliderOption(const char* string,
		Gwen::Controls::Base* base,
		float x, float y, GetterSetter<baseType, float> getter, float min, float max)
	{
		// Create label
		createLabel(string, base, x, y);

		// Create textbox
		SliderOption<baseType>* so =
			new SliderOption<baseType>(base, getter);
		so->getSlider()->SetPos(OPTIONS_X_START, y);
		so->getSlider()->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);
		so->getSlider()->SetRange(min, max);
		so->getSlider()->SetHeight(15);

		return so;
	}

	// Vector options
	template <typename baseType>
	VectorOption<baseType>* Tool::createVectorOption(const char* string,
		Gwen::Controls::Base* base,
		float x, float y, GetterSetter<baseType, b2Vec2> getter)
	{
		// Create label
		createLabel(string, base, x, y);

		// Create textbox
		VectorOption<baseType>* vo = 
			new VectorOption<baseType>(base, getter);
		vo->getTextBoxX()->SetPos(OPTIONS_X_START, y);
		vo->getTextBoxY()->SetPos(OPTIONS_X_START, y + LINE_HEIGHT);
		vo->getTextBoxX()->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);
		vo->getTextBoxY()->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);

		return vo;
	}

	// Int options
	template <typename baseType>
	IntOption<baseType>* Tool::createIntOption(const char* string,
		Gwen::Controls::Base* base,
		float x, float y, GetterSetter<baseType, int> getter)
	{
		// Create label
		createLabel(string, base, x, y);

		// Create textbox
		IntOption<baseType>* io = 
			new IntOption<baseType>(base, getter);
		io->getTextBox()->SetPos(OPTIONS_X_START, y);
		io->getTextBox()->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);

		return io;
	}

	// Multi options
	template <typename baseType, typename optionType>
	MultiOption<baseType, optionType>* Tool::createMultiOption(const char* string,
		Gwen::Controls::Base* base, float x, float y,
		GetterSetter<baseType, optionType> getter)
	{
		// Create label
		createLabel(string, base, x, y);

		// Create checkbox
		MultiOption<baseType, optionType>* mo = 
			new MultiOption<baseType, optionType>(base, getter);
		mo->getComboBox()->SetPos(OPTIONS_X_START, y);
		mo->getComboBox()->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);

		return mo;
	}
}

#endif /* VEHICLESIM_TOOL */
