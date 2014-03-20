#include "tools/Tool.h"

#include "VehicleSim.h"

#include <Gwen/Controls/Base.h>
#include <Gwen/Controls/ListBox.h>
#include <Gwen/Controls/TabButton.h>

#include "tools/tools/SelectionTool.h"

using namespace Gwen::Controls;

namespace vlr
{
	b2BodyDef Tool::_bodyDef;
	b2FixtureDef Tool::_fixtureDef;

	MotorInput Tool::_motorInput;
	std::vector<Gwen::Controls::Button*> Tool::_forwardButtons;
	std::vector<Gwen::Controls::Button*> Tool::_reverseButtons;
	
	b2Body* SelectionTool::_currentBody = nullptr;
	b2Joint* SelectionTool::_currentJoint = nullptr;

	Tool::Tool(VehicleSim* application, Layout::Tile* toolPanel,
		Base* optionsPanel, const char* icon, const char* name)
		: _app(application), _optionsPanel(optionsPanel),
		_dragging(false), _enabled(false), _options(nullptr)
	{
		_bodyDef.linearDamping = 0.5f;
		_bodyDef.angularDamping = 0.5f;
		_button = new Button(toolPanel);
		_button->SetIsToggle(true);
		_button->SetSize(36, 36);
		_button->SetImage(icon);
		_button->onToggleOn.Add(application, &VehicleSim::selectTool, this);
		_button->onToggleOff.Add(application, &VehicleSim::deselectTool);
		_button->SetToolTip(name);
		application->_tools.push_back(this);

		_fixtureDef.density = 1.0f;

		_toolOptions = new Base(optionsPanel);
		_toolOptions->Dock(Gwen::Pos::Fill);

		TabControl* tabControl = (TabControl*)optionsPanel;
		_tabButton = tabControl->AddPage(name, _toolOptions);
		hideOptions();

		_renderer = &application->_worldRenderer;
		_physWorld = &application->_physWorld;
		_camera = &application->_camera;
		_options = &application->_options;
		_statusLabel = application->_statusLabel;
	}

	void Tool::setEnabled(bool value)
	{
		if (value != _enabled)
		{
			if (value)
				onEnabled();
			else
				onDisabled();

			_enabled = value;
		}
	}

	void Tool::setText(const char* text)
	{
		if (_statusLabel != nullptr)
			_statusLabel->SetText(text);
	}

	void Tool::onEnabled()
	{

	}

	void Tool::onDisabled()
	{
		setText("");
	}

	void Tool::reset()
	{

	}

	void Tool::setSelected(bool val)
	{
		_button->SetToggleState(val);
	}

	bool Tool::getSelected() const
	{
		return _button->GetToggleState();
	}

	void Tool::resetOptions()
	{
		_bodyDef = b2BodyDef();
	}

	void Tool::update_base(float dt)
	{
		for (auto it = _updatableOptions.begin(); it != _updatableOptions.end(); ++it)
		{
			(*it)->update();
		}

		update(dt);
	}

	void Tool::update(float dt)
	{

	}

	void Tool::render()
	{

	}
}
