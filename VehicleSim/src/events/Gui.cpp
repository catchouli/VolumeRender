#include "VehicleSim.h"

// GWEN headers
#include <Gwen/Gwen.h>
#include <Gwen/Align.h>
#include <Gwen/Skins/Simple.h>
#include <Gwen/Skins/TexturedBase.h>
#include <Gwen/Input/Windows.h>
#include <Gwen/Renderers/OpenGL_DebugFont.h>

// Gwen controls
#include <Gwen/Controls/CollapsibleList.h>
#include <Gwen/Controls/DockBase.h>
#include <Gwen/Controls/ListBox.h>
#include <Gwen/Controls/MenuStrip.h>
#include <Gwen/Controls/RadioButton.h>
#include <Gwen/Controls/RadioButtonController.h>
#include <Gwen/Controls/TabControl.h>
#include <Gwen/Controls/TextBox.h>
#include <Gwen/Controls/Layout/Tile.h>
#include <Gwen/Controls/Layout/Position.h>

using namespace Gwen;
using namespace Controls;

namespace vlr
{
	void VehicleSim::initGui()
	{
		// Create GUI
		_guiDock = new Gwen::Controls::DockBase(_guiCanvas);
		_guiDock->Dock(Pos::Fill);

		// Create menu bar
		// TODO: set actions
		Gwen::Controls::MenuStrip* menu = new Gwen::Controls::MenuStrip(_guiCanvas);
		{
			Gwen::Controls::MenuItem* file = menu->AddItem("File");
			
			file->GetMenu()->AddItem("New", res::file::icon::NEW, "Ctrl + N");
			file->GetMenu()->AddItem("Open", res::file::icon::OPEN, "Ctrl + O");
			file->GetMenu()->AddItem("Save", res::file::icon::SAVE, "Ctrl + S");
			file->GetMenu()->AddItem("Save As", res::file::icon::NONE, "Ctrl + Shift + S");
			file->GetMenu()->AddItem("Exit", res::file::icon::NONE, "Ctrl + Q");
		}
		menu->SetWidth(getWidth());
		menu->Dock(Pos::Top);

		// Resize left panels of dock
		_guiDock->GetLeft()->SetWidth(94);

		// Create simulation panel
		Layout::Tile* simPanel = new Layout::Tile(_guiDock);
		_guiDock->GetLeft()->GetTop()->GetTabControl()->AddPage("Simulation", simPanel);
		simPanel->SetTileSize(38, 38);

		// Create sim button
		(new Button(simPanel))->SetSize(36, 36);
		(new Button(simPanel))->SetSize(36, 36);
		(new Button(simPanel))->SetSize(36, 36);
		_simButton = new Button(simPanel);
		_simButton->SetIsToggle(true);
		_simButton->SetSize(36, 36);
		_simButton->SetImage(res::file::icon::PLAY);
		_simButton->onToggleOn.Add(this, &VehicleSim::setSimulationRunning, new bool(true));
		_simButton->onToggleOff.Add(this, &VehicleSim::setSimulationRunning, new bool(false));
		
		// Create tool panel
		Layout::Tile* toolPanel = new Layout::Tile(_guiDock);
		toolPanel->SetTileSize(38, 38);
		_guiDock->GetLeft()->GetTabControl()->AddPage("Tools", toolPanel);

		Button* button;

		button = new Button(toolPanel, "Scroll");
		button->SetIsToggle(true);
		button->SetSize(36, 36);
		button->SetImage(res::file::icon::MOVE);
		button->onToggleOn.Add(this, &VehicleSim::selectTool);
		button->onToggleOff.Add(this, &VehicleSim::deselectTool);

		button = new Button(toolPanel, "Circle");
		button->SetIsToggle(true);
		button->SetSize(36, 36);
		button->SetImage(res::file::icon::CIRCLE);
		button->onToggleOn.Add(this, &VehicleSim::selectTool);
		button->onToggleOff.Add(this, &VehicleSim::deselectTool);

		// Create options panel
		_guiDock->GetRight()->GetTabControl()->AddPage("Options", new Base(_guiDock));
	}

	void VehicleSim::setSimulationRunning(Gwen::Event::Info info)
	{
		bool* data = (bool*)info.Data;

		_simulationRunning = *data;
	}

	void VehicleSim::selectTool(Gwen::Event::Info info)
	{
		// Deselect current tool if a tool is selected
		if (_currentToolButton != nullptr &&
			_currentToolButton != info.Control)
		{
			((Button*)_currentToolButton)->SetToggleState(false);
		}

		printf("%d\n", true);

		std::string tool = info.Control->GetName();
		
		if (tool == "Scroll")
			_currentTool = Tools::SCROLL;
		else if (tool == "Circle")
			_currentTool = Tools::CIRCLE;

		_currentToolButton = info.Control;

		printf("button selected: %s\n", info.Control->GetName().c_str());
	}

	void VehicleSim::deselectTool(Gwen::Event::Info info)
	{
		_currentTool = Tools::NONE;

		printf("button deselected: %s\n", info.Control->GetName().c_str());
	}
}
