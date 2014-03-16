#include "VehicleSim.h"

// GWEN headers
#include <Gwen/Gwen.h>
#include <Gwen/Align.h>
#include <Gwen/Skins/Simple.h>
#include <Gwen/Skins/TexturedBase.h>
#include <Gwen/Input/Windows.h>
#include <Gwen/Renderers/OpenGL_DebugFont.h>

// Gwen controls
#include <Gwen/ToolTip.h>
#include <Gwen/Controls/Button.h>
#include <Gwen/Controls/CollapsibleList.h>
#include <Gwen/Controls/DockBase.h>
#include <Gwen/Controls/GroupBox.h>
#include <Gwen/Controls/ListBox.h>
#include <Gwen/Controls/MenuStrip.h>
#include <Gwen/Controls/StatusBar.h>
#include <Gwen/Controls/RadioButton.h>
#include <Gwen/Controls/RadioButtonController.h>
#include <Gwen/Controls/TabControl.h>
#include <Gwen/Controls/TextBox.h>
#include <Gwen/Controls/Layout/Tile.h>
#include <Gwen/Controls/Layout/Position.h>

// Gwen dialogs
#include <Gwen/Controls/Dialogs/FileOpen.h>
#include <Gwen/Controls/Dialogs/FileSave.h>
#include <Gwen/Controls/Dialogs/Query.h>

// Custom controls
#include "tools/gui/FloatOption.h"
#include "tools/gui/IntOption.h"
#include "tools/gui/BoolOption.h"
#include "tools/gui/VectorOption.h"

// Tools
#include "tools/tools/MovementTool.h"
#include "tools/tools/SelectionTool.h"
#include "tools/tools/CircleTool.h"
#include "tools/tools/SquareTool.h"
#include "tools/tools/PolyTool.h"
#include "tools/tools/ZoomTool.h"
#include "tools/tools/RotateTool.h"

#include "tools/joints/DistanceTool.h"
#include "tools/joints/RevoluteTool.h"
#include "tools/joints/PrismaticTool.h"
#include "tools/joints/PulleyTool.h"
#include "tools/joints/WheelTool.h"
#include "tools/joints/WeldTool.h"

// Saving
#include "serialisation/Serialiser.h"

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
			
			file->GetMenu()->AddItem("New", res::file::icon::NONE)->SetAction(this, &VehicleSim::newDocument);
			file->GetMenu()->AddItem("Open", res::file::icon::NONE)->SetAction(this, &VehicleSim::loadDocument);
			file->GetMenu()->AddItem("Save", res::file::icon::NONE)->SetAction(this, &VehicleSim::saveDocument);
			file->GetMenu()->AddItem("Save As", res::file::icon::NONE)->SetAction(this, &VehicleSim::saveDocumentAs);
			file->GetMenu()->AddItem("Exit", res::file::icon::NONE)->SetAction(this, &VehicleSim::exitApplication);
		}
		menu->SetWidth(getWidth());
		menu->Dock(Pos::Top);

		// Create status bar
		Gwen::Controls::StatusBar* status = new Gwen::Controls::StatusBar(_guiCanvas);
		status->SetWidth(getWidth());
		_statusLabel = new Gwen::Controls::Label(status);
		_statusLabel->SetText("");
		_statusLabel->SetWidth(getWidth());
		status->AddControl(_statusLabel, false);
		status->Dock(Pos::Bottom);

		// Resize panels of dock
		_guiDock->GetLeft()->SetWidth(94);
		_guiDock->GetRight()->SetWidth(256);

		// Create simulation panel
		Layout::Tile* simPanel = new Layout::Tile(_guiDock);
		_guiDock->GetLeft()->GetTop()->GetTabControl()->AddPage("Simulation", simPanel);
		simPanel->SetTileSize(38, 38);
		_guiDock->GetLeft()->GetTop()->SetHeight(75);

		// Create sim button
		_simButton = new Button(simPanel);
		_simButton->SetIsToggle(true);
		_simButton->SetSize(36, 36);
		_simButton->SetImage(res::file::icon::PLAY);
		_simButton->onToggleOn.Add(this, &VehicleSim::setSimulationRunning, new bool(true));
		_simButton->onToggleOff.Add(this, &VehicleSim::setSimulationRunning, new bool(false));
		
		_simButton->SetToolTip("Toggle simulation");

		// Create reset button
		_resetButton = new Button(simPanel);
		_resetButton->SetSize(36, 36);
		_resetButton->SetImage(res::file::icon::RESET);
		_resetButton->onPress.Add(this, &VehicleSim::resetSimulation);
		_resetButton->SetDisabled(true);

		_resetButton->SetToolTip("Reset simulation");

		// Create tool panel
		Layout::Tile* toolPanel = new Layout::Tile(_guiDock);
		toolPanel->SetTileSize(38, 38);
		_guiDock->GetLeft()->GetTabControl()->AddPage("Tools", toolPanel);

		Layout::Tile* jointPanel = new Layout::Tile(_guiDock);
		jointPanel->SetTileSize(38, 38);
		_guiDock->GetLeft()->GetBottom()->GetTabControl()->AddPage("Joints", jointPanel);

		// Create world options
		Base* worldOptionsBase = new Base(_guiDock);
		_worldOptionsTabButton =
			_guiDock->GetRight()->GetTabControl()->AddPage("World Options", worldOptionsBase);

		{
			const float TEXTBOX_HEIGHT = 22;

			const float xstart = 100;
			const float xwidth = 240;
			float ypos = 0;

			// Create options
			Label* label;
			BoolOption<WorldOptions>* bo;

			// Gravity options
			label = new Label(worldOptionsBase);
			label->SetText("Gravity");
			label->SetPos(5, ypos);

			VectorOption<b2World>* vob2w =
				new VectorOption<b2World>(worldOptionsBase,
				&_physWorld, &b2World::GetGravity, &b2World::SetGravity);
			
			vob2w->getTextBoxX()->SetPos(xstart, ypos);
			vob2w->getTextBoxX()->SetWidth(xwidth - xstart);
			ypos += TEXTBOX_HEIGHT;

			vob2w->getTextBoxY()->SetPos(xstart, ypos);
			vob2w->getTextBoxY()->SetWidth(xwidth - xstart);
			ypos += TEXTBOX_HEIGHT;

			_updatableOptions.push_back(vob2w);

			// Drawing options
			ypos += 25;
			label = new Label(worldOptionsBase);
			label->SetText("Drawing options");
			label->SetPos(5, ypos);
			ypos += TEXTBOX_HEIGHT;

			// Draw shapes
			label = new Label(worldOptionsBase);
			label->SetText("  Draw shapes");
			label->SetPos(5, ypos);

			bo =
				new BoolOption<WorldOptions>(worldOptionsBase, &_worldOptions,
				&WorldOptions::getEnableDrawShapes, &WorldOptions::setEnableDrawShapes);
			_updatableOptions.push_back(bo);
			bo->getCheckBox()->SetPos(xstart, ypos);

			ypos += TEXTBOX_HEIGHT;

			// Draw joints
			label = new Label(worldOptionsBase);
			label->SetText("  Draw joints");
			label->SetPos(5, ypos);

			bo =
				new BoolOption<WorldOptions>(worldOptionsBase, &_worldOptions,
				&WorldOptions::getEnableDrawJoints, &WorldOptions::setEnableDrawJoints);
			_updatableOptions.push_back(bo);
			bo->getCheckBox()->SetPos(xstart, ypos);

			ypos += TEXTBOX_HEIGHT;

			// Draw AABBs
			label = new Label(worldOptionsBase);
			label->SetText("  Draw AABBs");
			label->SetPos(5, ypos);

			bo =
				new BoolOption<WorldOptions>(worldOptionsBase, &_worldOptions,
				&WorldOptions::getEnableDrawAABBs, &WorldOptions::setEnableDrawAABBs);
			_updatableOptions.push_back(bo);
			bo->getCheckBox()->SetPos(xstart, ypos);

			ypos += TEXTBOX_HEIGHT;

			// Simulation settings
			ypos += 25;
			label = new Label(worldOptionsBase);
			label->SetText("Simulation settings");
			label->SetPos(5, ypos);
			ypos += TEXTBOX_HEIGHT;

			// Steps per second
			label = new Label(worldOptionsBase);
			label->SetText("  Steps per second");
			label->SetPos(5, ypos);

			IntOption<VehicleSim>* iovs = 
				new IntOption<VehicleSim>(worldOptionsBase,
				this, &_timeStep);
			_updatableOptions.push_back(iovs);
			
			iovs->getTextBox()->SetPos(xstart, ypos);
			iovs->getTextBox()->SetWidth(xwidth - xstart);
			ypos += TEXTBOX_HEIGHT;

			// Velocity iterations
			label = new Label(worldOptionsBase);
			label->SetText("  Velocity iterations");
			label->SetPos(5, ypos);

			iovs = 
				new IntOption<VehicleSim>(worldOptionsBase,
				this, &_velocityIterations);
			_updatableOptions.push_back(iovs);
			
			iovs->getTextBox()->SetPos(xstart, ypos);
			iovs->getTextBox()->SetWidth(xwidth - xstart);
			ypos += TEXTBOX_HEIGHT;

			// Position iterations
			label = new Label(worldOptionsBase);
			label->SetText("  Position iterations");
			label->SetPos(5, ypos);

			iovs = 
				new IntOption<VehicleSim>(worldOptionsBase,
				this, &_positionIterations);
			_updatableOptions.push_back(iovs);
			
			iovs->getTextBox()->SetPos(xstart, ypos);
			iovs->getTextBox()->SetWidth(xwidth - xstart);
			ypos += TEXTBOX_HEIGHT;

			// Testing settings
			ypos += 25;
			label = new Label(worldOptionsBase);
			label->SetText("Testing settings");
			label->SetPos(5, ypos);
			ypos += TEXTBOX_HEIGHT;

			// Allow sleeping
			label = new Label(worldOptionsBase);
			label->SetText("  Allow sleeping");
			label->SetPos(5, ypos);

			BoolOption<b2World>* bob2w =
				new BoolOption<b2World>(worldOptionsBase, &_physWorld,
				&b2World::GetAllowSleeping, &b2World::SetAllowSleeping);
			_updatableOptions.push_back(bob2w);
			bob2w->getCheckBox()->SetPos(xstart, ypos);
			ypos += TEXTBOX_HEIGHT;

			// Warm starting
			label = new Label(worldOptionsBase);
			label->SetText("  Warm starting");
			label->SetPos(5, ypos);

			bob2w =
				new BoolOption<b2World>(worldOptionsBase, &_physWorld,
				&b2World::GetWarmStarting, &b2World::SetWarmStarting);
			_updatableOptions.push_back(bob2w);
			bob2w->getCheckBox()->SetPos(xstart, ypos);
			ypos += TEXTBOX_HEIGHT;

			// Continuous physics
			label = new Label(worldOptionsBase);
			label->SetText("  Continuous physics");
			label->SetPos(5, ypos);

			bob2w =
				new BoolOption<b2World>(worldOptionsBase, &_physWorld,
				&b2World::GetContinuousPhysics, &b2World::SetContinuousPhysics);
			_updatableOptions.push_back(bob2w);
			bob2w->getCheckBox()->SetPos(xstart, ypos);
			ypos += TEXTBOX_HEIGHT;

			// Sub stepping
			label = new Label(worldOptionsBase);
			label->SetText("  Sub stepping");
			label->SetPos(5, ypos);

			bob2w =
				new BoolOption<b2World>(worldOptionsBase, &_physWorld,
				&b2World::GetSubStepping, &b2World::SetSubStepping);
			_updatableOptions.push_back(bob2w);
			bob2w->getCheckBox()->SetPos(xstart, ypos);
		}

		// Create tools
		new MovementTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::MOVE);
		new SelectionTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::SELECT);
		new CircleTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::CIRCLE);
		new RotateTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::ROTATE);
		new SquareTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::SQUARE);
		new PolyTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::POLY);
		new ZoomTool(this, toolPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::ZOOM);

		new DistanceTool(this, jointPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::DISTANCE);
		new RevoluteTool(this, jointPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::REVOLUTE);
		new PrismaticTool(this, jointPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::PRISMATIC);
		new PulleyTool(this, jointPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::PULLY);
		new WheelTool(this, jointPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::WHEEL);
		new WeldTool(this, jointPanel, _guiDock->GetRight()->GetTabControl(), res::file::icon::WELD);
		
		// Select world options by default
		_worldOptionsTabButton->DoAction();
	}

	void VehicleSim::setSimulationRunning(Gwen::Event::Info info)
	{
		bool* data = (bool*)info.Data;

		_simulationRunning = *data;

		if (_simulationRunning)
		{
			// Store state
			_storedState = Serialiser::serialiseWorld(&_physWorld);
			_resetButton->SetDisabled(false);
		}
		else
		{
			// Restore state
			if (_currentTool != nullptr)
				_currentTool->reset();
			Serialiser::destroyWorld(&_physWorld);
			Serialiser::deserialiseWorld(&_physWorld, _storedState);
			_resetButton->SetDisabled(true);
		}
	}

	void VehicleSim::resetSimulation(Gwen::Event::Info info)
	{
		if (_currentTool != nullptr)
			_currentTool->reset();

		Serialiser::destroyWorld(&_physWorld);
		Serialiser::deserialiseWorld(&_physWorld, _storedState);
	}

	void VehicleSim::selectTool(Gwen::Event::Info info)
	{
		// Deselect current tool if a tool is selected
		if (_currentToolButton != nullptr)
		{
			((Button*)_currentToolButton)->SetToggleState(false);
		}

		// Get tool name
		_currentTool = (Tool*)info.Data;
		_currentTool->setEnabled(true);
		_currentTool->showOptions();

		// Set current tool button
		_currentToolButton = info.Control;
	}

	void VehicleSim::deselectTool(Gwen::Event::Info info)
	{
		// Reset tool state
		_currentTool->reset();
		_currentTool->setEnabled(false);
		_currentTool->hideOptions();
		_currentToolButton = nullptr;

		// Show world options
		_worldOptionsTabButton->DoAction();

		// Set new tool
		_currentTool = nullptr;
	}

	void VehicleSim::disableSim()
	{
		// Disable simulation if running
		if (_simulationRunning)
		{
			_simButton->Toggle();
		}
	}

	void VehicleSim::getString(Gwen::Event::Info info)
	{
		_gotString = true;
		_lastString = info.String.c_str();
	}

	void VehicleSim::newDocument(Gwen::Event::Info info)
	{
		if (_currentTool != nullptr)
			_currentTool->reset();

		// Reset world
		Serialiser::destroyWorld(&_physWorld);

		// Disable simulation
		disableSim();
		doStep();

		_filename = "";
	}

	void VehicleSim::saveDocument(Gwen::Event::Info info)
	{
		// Disable simulation
		disableSim();
		doStep();

		// Serialize world
		std::string world = Serialiser::serialiseWorld(&_physWorld);

		// Try to save file as current filename if set
		if (_filename != "")
		{
			FILE* file = fopen(_filename.c_str(), "wb");

			if (file == 0)
			{
				saveDocumentAs(info);
				return;
			}

			size_t written = fwrite(world.c_str(), 1, world.length(), file);
			fclose(file);

			if (written <= 0)
			{
				saveDocumentAs(info);
			}
		}
		else
		{
			saveDocumentAs(info);
		}
	}

	void VehicleSim::saveDocumentAs(Gwen::Event::Info info)
	{
		// Disable simulation
		disableSim();
		doStep();		

		// Serialize world
		std::string world = Serialiser::serialiseWorld(&_physWorld);

		// Get file path and name from user
		_gotString = false;
		Gwen::Dialogs::FileSave(true, "Save simulation", ".", "Simulated world (*.sim) | *.sim", this, &VehicleSim::getString);

		// Save file
		if (_gotString)
		{
			_filename = _lastString;

			// Save document
			FILE* file = fopen(_filename.c_str(), "wb");

			if (file == 0)
			{
				saveDocumentAs(info);
			}

			size_t wrote = fwrite(world.c_str(), 1, world.length(), file);
			fclose(file);

			if (wrote <= 0)
				saveDocumentAs(info);
		}
	}

	void VehicleSim::loadDocument(Gwen::Event::Info info)
	{
		if (_currentTool != nullptr)
			_currentTool->reset();

		// Reset world
		Serialiser::destroyWorld(&_physWorld);

		// Disable simulation
		disableSim();

		// Get file path and name from user
		_gotString = false;
		Gwen::Dialogs::FileOpen(true, "Load simulation", ".", "Simulated world (*.sim) | *.sim", this, &VehicleSim::getString);

		// Load file
		if (_gotString)
		{
			long len;
			char* buffer;

			// Load document
			FILE* file = fopen(_lastString.c_str(), "rb");

			if (file == 0)
			{
				loadDocument(info);
				return;
			}

			// Get file length
			fseek(file, 0, SEEK_END);
			len = ftell(file);
			fseek(file, 0, SEEK_SET);

			// Load data
			buffer = (char*)malloc(len);
			size_t read = fread(buffer, 1, len, file);
			fclose(file);

			if (read <= 0)
			{
				free(buffer);
				loadDocument(info);
			}

			// Parse it to json
			Serialiser::deserialiseWorld(&_physWorld, buffer);

			free(buffer);
			_filename = _lastString;
		}
	}

	void VehicleSim::exitApplication(Gwen::Event::Info info)
	{
		end();
	}
}
