#include "tools/Tool.h"

#include <Gwen/Controls/Label.h>

#include <Box2D/Box2D.h>

#include "tools/tools/SelectionTool.h"

#include "tools/joints/DistanceTool.h"
#include "tools/joints/PrismaticTool.h"
#include "tools/joints/PulleyTool.h"
#include "tools/joints/RevoluteTool.h"
#include "tools/joints/WeldTool.h"
#include "tools/joints/WheelTool.h"

#include "tools/gui/FloatOption.h"
#include "tools/gui/IntOption.h"
#include "tools/gui/BoolOption.h"
#include "tools/gui/MultiOption.h"
#include "tools/gui/VectorOption.h"
#include "tools/gui/SliderOption.h"

#include "VehicleSim.h"

using namespace Gwen;
using namespace Gwen::Controls;

namespace vlr
{
	Gwen::Controls::Base* SelectionTool::_jointsPage = nullptr;
	Gwen::Controls::Base* SelectionTool::_jointOptions = nullptr;
	Gwen::Controls::Base* SelectionTool::_jointOptionsPanel = nullptr;
	Gwen::Controls::ListBox* SelectionTool::_listBox = nullptr;
	Gwen::Controls::TabButton* SelectionTool::_jointsButton = nullptr;

	Gwen::Controls::DockBase* Tool::getDock()
	{
		return _app->_guiDock;
	}

	void Tool::createNoOptions()
	{
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetText("No options available");
	}

	float Tool::createBodyGui(Gwen::Controls::Base* parent, float ypos)
	{
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		// Body options
		Label* label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Body options");
		ypos += LINE_HEIGHT;

		// Body type
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Body type");

		// Body type
		MultiOption<b2BodyDef, b2BodyType>* mo =
			new MultiOption<b2BodyDef, b2BodyType>(parent, &_bodyDef, &_bodyDef.type);
		mo->addOption("Static", b2_staticBody);
		mo->addOption("Dynamic", b2_dynamicBody);
		mo->getComboBox()->SetPos(xstart, ypos);
		mo->getComboBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(mo);
		ypos += LINE_HEIGHT;
		
		// Active
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Active");

		BoolOption<b2BodyDef>* bo = 
			new BoolOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.active);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Allow sleep
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Allow sleep");

		bo = 
			new BoolOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.allowSleep);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Allow sleep
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Awake");

		bo = 
			new BoolOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.awake);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Fix rotation
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Fix rotation");

		bo = 
			new BoolOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.fixedRotation);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Is fast moving
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Is fast moving");

		bo = 
			new BoolOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.bullet);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Gravity Scale
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Gravity scale");

		FloatOption<b2BodyDef>* fo = 
			new FloatOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.gravityScale);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Initial velocities
		label = new Label(parent);
		label->SetText("  Initial velocities");
		label->SetPos(5, ypos);
		ypos += LINE_HEIGHT;

		// Linear
		label = new Label(parent);
		label->SetText("  Linear");
		label->SetPos(5, ypos);

		VectorOption<b2BodyDef>* vo = 
			new VectorOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.linearVelocity);
		vo->getTextBoxX()->SetPos(xstart, ypos);
		vo->getTextBoxY()->SetPos(xstart, ypos + LINE_HEIGHT);
		vo->getTextBoxX()->SetWidth(xwidth - xstart);
		vo->getTextBoxY()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(vo);

		ypos += LINE_HEIGHT * 2;

		// Angular
		label = new Label(parent);
		label->SetText("  Angular");
		label->SetPos(5, ypos);

		fo = 
			new FloatOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.angularVelocity);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Velocity Damping
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Velocity Damping");

		ypos += LINE_HEIGHT;

		// Linear
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Linear");

		SliderOption<b2BodyDef>* so =
			new SliderOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.linearDamping);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT;

		// Angular
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Angular");

		so =
			new SliderOption<b2BodyDef>(parent,
			&_bodyDef, &_bodyDef.angularDamping);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetHeight(15);
		so->getSlider()->SetRange(0.0f, 1.0f);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT * 3;

		return ypos;
	}
	
	float Tool::createFixtureGui(Gwen::Controls::Base* parent, float ypos)
	{
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		// Fixture options
		Label* label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Fixture options");
		ypos += LINE_HEIGHT;
		
		// Friction
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Friction");
		
		SliderOption<b2FixtureDef>* so =
			new SliderOption<b2FixtureDef>(parent,
			&_fixtureDef, &_fixtureDef.friction);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT;
		
		// Restitution
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Restitution");
		
		so =
			new SliderOption<b2FixtureDef>(parent,
			&_fixtureDef, &_fixtureDef.restitution);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT;
		
		// Density
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Density");

		FloatOption<b2FixtureDef>* fo = 
			new FloatOption<b2FixtureDef>(parent,
			&_fixtureDef, &_fixtureDef.density);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		return ypos;
	}

	float Tool::createJointInputGui(Gwen::Controls::Base* parent, float ypos)
	{
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		// Input options
		Label* label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Input Options");

		ypos += LINE_HEIGHT;

		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Max force");

		FloatOption<MotorInput>* fo = 
			new FloatOption<MotorInput>(parent,
			&_motorInput, &_motorInput.maxForce);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Speed");

		fo = 
			new FloatOption<MotorInput>(parent,
			&_motorInput, &_motorInput.speed);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Forward key");

		Gwen::Controls::Button* _forwardButton = new Button(parent);
		_forwardButton->SetPos(xstart, ypos);
		_forwardButton->SetWidth(xwidth - xstart);
		_forwardButton->SetText(InputConverter::translateCharToString(_motorInput.forwardButton));
		_forwardButton->UserData.Set("updateKey", &_motorInput.forwardButton);
		_forwardButton->SetIsToggle(true);

		_forwardButtons.push_back(_forwardButton);

		ypos += LINE_HEIGHT;

		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Reverse key");

		Gwen::Controls::Button* _reverseButton = new Button(parent);
		_reverseButton->SetPos(xstart, ypos);
		_reverseButton->SetWidth(xwidth - xstart);
		_reverseButton->SetText(InputConverter::translateCharToString(_motorInput.reverseButton));
		_reverseButton->UserData.Set("updateKey", &_motorInput.reverseButton);
		_reverseButton->SetIsToggle(true);

		_reverseButtons.push_back(_reverseButton);

		return ypos;
	}

	float SelectionTool::createJointInputGuiSelection(
		Gwen::Controls::Base* parent, float ypos, b2Joint* joint)
	{
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		MotorInput* motorInput = (MotorInput*)joint->GetUserData();

		if (motorInput != nullptr)
		{
			// Input options
			Label* label = new Label(parent);
			label->SetPos(5, ypos);
			label->SetText("Input Options");

			ypos += LINE_HEIGHT;

			label = new Label(parent);
			label->SetPos(5, ypos);
			label->SetText("  Max force");

			FloatOption<MotorInput>* fo = 
				new FloatOption<MotorInput>(parent,
				motorInput, &motorInput->maxForce);
			fo->getTextBox()->SetPos(xstart, ypos);
			fo->getTextBox()->SetWidth(xwidth - xstart);
			_jointOptionsUpdatables.push_back(fo);

			ypos += LINE_HEIGHT;

			label = new Label(parent);
			label->SetPos(5, ypos);
			label->SetText("  Speed");

			fo = 
				new FloatOption<MotorInput>(parent,
				motorInput, &motorInput->speed);
			fo->getTextBox()->SetPos(xstart, ypos);
			fo->getTextBox()->SetWidth(xwidth - xstart);
			_jointOptionsUpdatables.push_back(fo);

			ypos += LINE_HEIGHT;

			label = new Label(parent);
			label->SetPos(5, ypos);
			label->SetText("  Forward key");

			Gwen::Controls::Button* _forwardButton = new Button(parent);
			_forwardButton->SetPos(xstart, ypos);
			_forwardButton->SetWidth(xwidth - xstart);
			_forwardButton->SetText(InputConverter::translateCharToString(motorInput->forwardButton));
			_forwardButton->UserData.Set("updateKey", &motorInput->forwardButton);
			_forwardButton->SetIsToggle(true);

			_inputButtons.push_back(_forwardButton);

			ypos += LINE_HEIGHT;

			label = new Label(parent);
			label->SetPos(5, ypos);
			label->SetText("  Reverse key");

			Gwen::Controls::Button* _reverseButton = new Button(parent);
			_reverseButton->SetPos(xstart, ypos);
			_reverseButton->SetWidth(xwidth - xstart);
			_reverseButton->SetText(InputConverter::translateCharToString(motorInput->reverseButton));
			_reverseButton->UserData.Set("updateKey", &motorInput->reverseButton);
			_reverseButton->SetIsToggle(true);

			_inputButtons.push_back(_reverseButton);
		}

		return ypos;
	}

	void SelectionTool::initGui()
	{
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		Gwen::Controls::Base* parent = _otherOptions;

		// Body options
		Label* label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Body options");
		ypos += LINE_HEIGHT;

		// Body type
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Body type");

		// Body type
		MultiOption<b2Body, b2BodyType>* mo =
			new MultiOption<b2Body, b2BodyType>(parent, nullptr,
			&b2Body::GetType, &b2Body::SetType);
		mo->addOption("Static", b2_staticBody);
		mo->addOption("Dynamic", b2_dynamicBody);
		mo->getComboBox()->SetPos(xstart, ypos);
		mo->getComboBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(mo);
		_bodyOptions.push_back(mo);
		ypos += LINE_HEIGHT;
		
		// Active
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Active");

		BoolOption<b2Body>* bo = 
			new BoolOption<b2Body>(parent,
			nullptr, &b2Body::IsActive, &b2Body::SetActive);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Allow sleep
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Allow sleep");

		bo = 
			new BoolOption<b2Body>(parent,
			nullptr, &b2Body::IsSleepingAllowed, &b2Body::SetSleepingAllowed);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Allow sleep
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Awake");

		bo = 
			new BoolOption<b2Body>(parent,
			nullptr, &b2Body::IsAwake, &b2Body::SetAwake);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Fix rotation
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Fix rotation");

		bo = 
			new BoolOption<b2Body>(parent,
			nullptr, &b2Body::IsFixedRotation, &b2Body::SetFixedRotation);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Is fast moving
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Is fast moving");

		bo = 
			new BoolOption<b2Body>(parent,
			nullptr, &b2Body::IsBullet, &b2Body::SetBullet);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Gravity Scale
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Gravity scale");

		FloatOption<b2Body>* fo = 
			new FloatOption<b2Body>(parent,
			nullptr, &b2Body::GetGravityScale, &b2Body::SetGravityScale);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		_bodyOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Initial velocities
		label = new Label(parent);
		label->SetText("  Initial velocities");
		label->SetPos(5, ypos);
		ypos += LINE_HEIGHT;

		// Linear
		label = new Label(parent);
		label->SetText("  Linear");
		label->SetPos(5, ypos);

		VectorOption<b2Body>* vo = 
			new VectorOption<b2Body>(parent,
			nullptr, &b2Body::GetLinearVelocity, &b2Body::SetLinearVelocity);
		vo->getTextBoxX()->SetPos(xstart, ypos);
		vo->getTextBoxY()->SetPos(xstart, ypos + LINE_HEIGHT);
		vo->getTextBoxX()->SetWidth(xwidth - xstart);
		vo->getTextBoxY()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(vo);
		_bodyOptions.push_back(vo);

		ypos += LINE_HEIGHT * 2;

		// Angular
		label = new Label(parent);
		label->SetText("  Angular");
		label->SetPos(5, ypos);

		fo = 
			new FloatOption<b2Body>(parent,
			nullptr, &b2Body::GetAngularVelocity, &b2Body::SetAngularVelocity);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		_bodyOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Velocity Damping
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Velocity Damping");

		ypos += LINE_HEIGHT;

		// Linear
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Linear");

		SliderOption<b2Body>* so =
			new SliderOption<b2Body>(parent,
			nullptr, &b2Body::GetLinearDamping, &b2Body::SetLinearDamping);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);
		_bodyOptions.push_back(so);

		ypos += LINE_HEIGHT;

		// Angular
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Angular");

		so =
			new SliderOption<b2Body>(parent,
			nullptr, &b2Body::GetAngularDamping, &b2Body::SetAngularDamping);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetHeight(15);
		so->getSlider()->SetRange(0.0f, 1.0f);
		_updatableOptions.push_back(so);
		_bodyOptions.push_back(so);

		ypos += LINE_HEIGHT * 3;

		// Fixture options
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("Fixture options");
		ypos += LINE_HEIGHT;
		
		// Friction
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Friction");
		
		SliderOption<b2Fixture>* sob2f =
			new SliderOption<b2Fixture>(parent,
			nullptr, &b2Fixture::GetFriction, &b2Fixture::SetFriction);
		sob2f->getSlider()->SetPos(xstart, ypos);
		sob2f->getSlider()->SetWidth(xwidth - xstart);
		sob2f->getSlider()->SetRange(0.0f, 1.0f);
		sob2f->getSlider()->SetHeight(15);
		_updatableOptions.push_back(sob2f);
		_fixtureOptions.push_back(sob2f);

		ypos += LINE_HEIGHT;
		
		// Restitution
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Restitution");
		
		sob2f =
			new SliderOption<b2Fixture>(parent,
			nullptr, &b2Fixture::GetRestitution, &b2Fixture::SetRestitution);
		sob2f->getSlider()->SetPos(xstart, ypos);
		sob2f->getSlider()->SetWidth(xwidth - xstart);
		sob2f->getSlider()->SetRange(0.0f, 1.0f);
		sob2f->getSlider()->SetHeight(15);
		_updatableOptions.push_back(sob2f);
		_fixtureOptions.push_back(sob2f);

		ypos += LINE_HEIGHT;
		
		// Density
		label = new Label(parent);
		label->SetPos(5, ypos);
		label->SetText("  Density");

		FloatOption<b2Fixture>* fob2f = 
			new FloatOption<b2Fixture>(parent,
			nullptr, &b2Fixture::GetDensity, &b2Fixture::SetDensity);
		fob2f->getTextBox()->SetPos(xstart, ypos);
		fob2f->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fob2f);
		_fixtureOptions.push_back(fob2f);
		ypos += LINE_HEIGHT;
	}

	void SelectionTool::selectJoint(b2Joint* joint)
	{
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 10;

		delete _jointOptions;
		_jointOptions = new Base(_jointOptionsPanel);
		_jointOptions->Dock(Pos::Fill);

		auto label = new Label(_jointOptions);
		label->SetText("Joint options");
		label->SetPos(5, ypos);
		ypos += LINE_HEIGHT;

		switch (joint->GetType())
		{
		case b2JointType::e_distanceJoint:
			{
				b2DistanceJoint* specJoint = (b2DistanceJoint*)joint;

				// Length
				Gwen::Controls::Label* label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Length");

				FloatOption<b2DistanceJoint>* fo = 
					new FloatOption<b2DistanceJoint>(_jointOptions,
					specJoint, &b2DistanceJoint::GetLength, &b2DistanceJoint::SetLength);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Frequency
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Frequency");

				fo = 
					new FloatOption<b2DistanceJoint>(_jointOptions,
					specJoint, &b2DistanceJoint::GetFrequency, &b2DistanceJoint::SetFrequency);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Damping ratio
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Damping ratio");
		
				SliderOption<b2DistanceJoint>* so =
					new SliderOption<b2DistanceJoint>(_jointOptions,
					specJoint, &b2DistanceJoint::GetDampingRatio, &b2DistanceJoint::SetDampingRatio);
				so->getSlider()->SetPos(xstart, ypos);
				so->getSlider()->SetWidth(xwidth - xstart);
				so->getSlider()->SetRange(0.0f, 1.0f);
				so->getSlider()->SetHeight(15);
				_jointOptionsUpdatables.push_back(so);

				ypos += LINE_HEIGHT * 2;
			}
			break;
		case b2JointType::e_prismaticJoint:
			{
				b2PrismaticJoint* specJoint = (b2PrismaticJoint*)joint;
				
				// Collide with connected
				Gwen::Controls::Label* label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Collide connected");

				BoolOption<b2PrismaticJoint>* bo = 
					new BoolOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::GetCollideConnected, nullptr);
				bo->getCheckBox()->SetPos(xstart, ypos);
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Reference angle
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Reference angle");

				FloatOption<b2PrismaticJoint>* fo = 
					new FloatOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::GetReferenceAngle, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Enable limit
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Enable limits");

				bo = 
					new BoolOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::IsLimitEnabled, &b2PrismaticJoint::EnableLimit);
				bo->getCheckBox()->SetPos(xstart, ypos);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Lower limit
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Lower limit");

				fo = 
					new FloatOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::GetLowerLimit, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Upper limit
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Upper limit");

				fo = 
					new FloatOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::GetUpperLimit, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Enable motor
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Enable motor");

				bo = 
					new BoolOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::IsMotorEnabled, &b2PrismaticJoint::EnableMotor);
				bo->getCheckBox()->SetPos(xstart, ypos);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Max force
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Max motor force");

				fo = 
					new FloatOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::GetMaxMotorForce, &b2PrismaticJoint::SetMaxMotorForce);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Motor speed
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Motor speed");

				fo = 
					new FloatOption<b2PrismaticJoint>(_jointOptions,
					specJoint, &b2PrismaticJoint::GetMotorSpeed, &b2PrismaticJoint::SetMotorSpeed);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT * 2;
			}
			break;
		case b2JointType::e_pulleyJoint:
			{
				b2PulleyJoint* specJoint = (b2PulleyJoint*)joint;

				// Collide with connected
				Gwen::Controls::Label* label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Collide connected");

				BoolOption<b2PulleyJoint>* bo = 
					new BoolOption<b2PulleyJoint>(_jointOptions,
					specJoint, &b2PulleyJoint::GetCollideConnected, nullptr);
				bo->getCheckBox()->SetPos(xstart, ypos);
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Length A
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Length A");

				FloatOption<b2PulleyJoint>* fo = 
					new FloatOption<b2PulleyJoint>(_jointOptions,
					specJoint, &b2PulleyJoint::GetLengthA, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Length B
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Length B");

				fo = 
					new FloatOption<b2PulleyJoint>(_jointOptions,
					specJoint, &b2PulleyJoint::GetLengthB, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Ratio
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Ratio");

				fo = 
					new FloatOption<b2PulleyJoint>(_jointOptions,
					specJoint, &b2PulleyJoint::GetRatio, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT * 2;
			}
			break;
		case b2JointType::e_revoluteJoint:
			{
				b2RevoluteJoint* specJoint = (b2RevoluteJoint*)joint;

				// Collide with connected
				Gwen::Controls::Label* label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Collide connected");

				BoolOption<b2RevoluteJoint>* bo = 
					new BoolOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::GetCollideConnected, nullptr);
				bo->getCheckBox()->SetPos(xstart, ypos);
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Reference angle
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Reference angle");

				FloatOption<b2RevoluteJoint>* fo = 
					new FloatOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::GetReferenceAngle, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Enable limit
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Enable limits");

				bo = 
					new BoolOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::IsLimitEnabled, &b2RevoluteJoint::EnableLimit);
				bo->getCheckBox()->SetPos(xstart, ypos);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Lower limit
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Lower angle");

				fo = 
					new FloatOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::GetLowerLimit, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Upper limit
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Upper angle");

				fo = 
					new FloatOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::GetUpperLimit, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Enable motor
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Enable motor");

				bo = 
					new BoolOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::IsMotorEnabled, &b2RevoluteJoint::EnableMotor);
				bo->getCheckBox()->SetPos(xstart, ypos);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Max force
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Max motor torque");

				fo = 
					new FloatOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::GetMaxMotorTorque, &b2RevoluteJoint::SetMaxMotorTorque);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Motor speed
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Motor speed");

				fo = 
					new FloatOption<b2RevoluteJoint>(_jointOptions,
					specJoint, &b2RevoluteJoint::GetMotorSpeed, &b2RevoluteJoint::SetMotorSpeed);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT * 2;
			}
			break;
		case b2JointType::e_weldJoint:
			{
				b2WeldJoint* specJoint = (b2WeldJoint*)joint;

				// Collide with connected
				Gwen::Controls::Label* label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Collide connected");

				BoolOption<b2WeldJoint>* bo = 
					new BoolOption<b2WeldJoint>(_jointOptions,
					specJoint, &b2WeldJoint::GetCollideConnected, nullptr);
				bo->getCheckBox()->SetPos(xstart, ypos);
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Reference angle
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Reference angle");

				FloatOption<b2WeldJoint>* fo = 
					new FloatOption<b2WeldJoint>(_jointOptions,
					specJoint, &b2WeldJoint::GetReferenceAngle, nullptr);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Frequency
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Frequency");

				fo = 
					new FloatOption<b2WeldJoint>(_jointOptions,
					specJoint, &b2WeldJoint::GetFrequency, &b2WeldJoint::SetFrequency);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Damping ratio
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Damping ratio");
		
				SliderOption<b2WeldJoint>* so =
					new SliderOption<b2WeldJoint>(_jointOptions,
					specJoint, &b2WeldJoint::GetDampingRatio, &b2WeldJoint::SetDampingRatio);
				so->getSlider()->SetPos(xstart, ypos);
				so->getSlider()->SetWidth(xwidth - xstart);
				so->getSlider()->SetRange(0.0f, 1.0f);
				so->getSlider()->SetHeight(15);
				_jointOptionsUpdatables.push_back(so);

				ypos += LINE_HEIGHT * 2;
			}
			break;
		case b2JointType::e_wheelJoint:
			{
				b2WheelJoint* specJoint = (b2WheelJoint*)joint;

				// Collide with connected
				Gwen::Controls::Label* label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Collide connected");

				BoolOption<b2WheelJoint>* bo = 
					new BoolOption<b2WheelJoint>(_jointOptions,
					specJoint, &b2WheelJoint::GetCollideConnected, nullptr);
				bo->getCheckBox()->SetPos(xstart, ypos);
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Enable motor
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Enable motor");

				bo = 
					new BoolOption<b2WheelJoint>(_jointOptions,
					specJoint, &b2WheelJoint::IsMotorEnabled, &b2WheelJoint::EnableMotor);
				bo->getCheckBox()->SetPos(xstart, ypos);
				_jointOptionsUpdatables.push_back(bo);

				ypos += LINE_HEIGHT;

				// Max force
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Max motor torque");

				FloatOption<b2WheelJoint>* fo = 
					new FloatOption<b2WheelJoint>(_jointOptions,
					specJoint, &b2WheelJoint::GetMaxMotorTorque, &b2WheelJoint::SetMaxMotorTorque);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Motor speed
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("  Motor speed");

				fo = 
					new FloatOption<b2WheelJoint>(_jointOptions,
					specJoint, &b2WheelJoint::GetMotorSpeed, &b2WheelJoint::SetMotorSpeed);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);

				ypos += LINE_HEIGHT;

				// Frequency
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Frequency");

				fo = 
					new FloatOption<b2WheelJoint>(_jointOptions,
					specJoint, &b2WheelJoint::GetSpringFrequencyHz, &b2WheelJoint::SetSpringFrequencyHz);
				fo->getTextBox()->SetPos(xstart, ypos);
				fo->getTextBox()->SetWidth(xwidth - xstart);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Damping ratio
				label = new Gwen::Controls::Label(_jointOptions);
				label->SetPos(5, ypos);
				label->SetText("Damping ratio");
		
				SliderOption<b2WheelJoint>* so =
					new SliderOption<b2WheelJoint>(_jointOptions,
					specJoint, &b2WheelJoint::GetSpringDampingRatio, &b2WheelJoint::SetSpringDampingRatio);
				so->getSlider()->SetPos(xstart, ypos);
				so->getSlider()->SetWidth(xwidth - xstart);
				so->getSlider()->SetRange(0.0f, 1.0f);
				so->getSlider()->SetHeight(15);
				_jointOptionsUpdatables.push_back(so);

				ypos += LINE_HEIGHT * 2;
			}
			break;
		}

		ypos = createJointInputGuiSelection(_jointOptions, ypos, _currentJoint);

		ypos += LINE_HEIGHT * 2;

		auto button = new Gwen::Controls::Button(_jointOptions);
		button->SetPos(5, ypos);
		button->SetText("Delete Joint");
		button->onPress.Add(this, &SelectionTool::deleteJoint);
	}

	void DistanceTool::initGui()
	{
		// Create GUI
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		// Collide with connected
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Collide connected");

		BoolOption<b2DistanceJointDef>* bo = 
			new BoolOption<b2DistanceJointDef>(_toolOptions,
			&_distanceJointDef, &_distanceJointDef.collideConnected);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Frequency
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Frequency");

		FloatOption<b2DistanceJointDef>* fo = 
			new FloatOption<b2DistanceJointDef>(_toolOptions,
			&_distanceJointDef, &_distanceJointDef.frequencyHz);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Damping ratio
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Damping ratio");
		
		SliderOption<b2DistanceJointDef>* so =
			new SliderOption<b2DistanceJointDef>(_toolOptions,
			&_distanceJointDef, &_distanceJointDef.dampingRatio);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT;
	}

	void PrismaticTool::initGui()
	{
		// Create GUI
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		// Collide with connected
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Collide connected");

		BoolOption<b2PrismaticJointDef>* bo = 
			new BoolOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.collideConnected);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Reference angle
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Reference angle");

		FloatOption<b2PrismaticJointDef>* fo = 
			new FloatOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.referenceAngle);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Enable limit
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Enable limits");

		bo = 
			new BoolOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.enableLimit);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Lower limit
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Lower limit");

		fo = 
			new FloatOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.lowerTranslation);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Upper limit
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Upper limit");

		fo = 
			new FloatOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.upperTranslation);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Enable motor
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Enable motor");

		bo = 
			new BoolOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.enableMotor);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Max force
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Max motor force");

		fo = 
			new FloatOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.maxMotorForce);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Motor speed
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Motor speed");

		fo = 
			new FloatOption<b2PrismaticJointDef>(_toolOptions,
			&_prismaticJointDef, &_prismaticJointDef.motorSpeed);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		ypos = createJointInputGui(_toolOptions, ypos);
	}

	void PulleyTool::initGui()
	{
		// Create GUI
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		// Collide with connected
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Collide connected");

		BoolOption<b2PulleyJointDef>* bo = 
			new BoolOption<b2PulleyJointDef>(_toolOptions,
			&_pulleyJointDef, &_pulleyJointDef.collideConnected);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Ratio
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Ratio");

		FloatOption<b2PulleyJointDef>* fo = 
			new FloatOption<b2PulleyJointDef>(_toolOptions,
			&_pulleyJointDef, &_pulleyJointDef.ratio);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
	}

	void RevoluteTool::initGui()
	{
		// Create GUI
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		// Collide with connected
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Collide connected");

		BoolOption<b2RevoluteJointDef>* bo = 
			new BoolOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.collideConnected);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Reference angle
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Reference angle");

		FloatOption<b2RevoluteJointDef>* fo = 
			new FloatOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.referenceAngle);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Enable limit
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Enable limits");

		bo = 
			new BoolOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.enableLimit);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Lower limit
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Lower angle");

		fo = 
			new FloatOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.lowerAngle);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Upper limit
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Upper angle");

		fo = 
			new FloatOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.upperAngle);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Enable motor
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Enable motor");

		bo = 
			new BoolOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.enableMotor);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Max force
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Max motor torque");

		fo = 
			new FloatOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.maxMotorTorque);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Motor speed
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Motor speed");

		fo = 
			new FloatOption<b2RevoluteJointDef>(_toolOptions,
			&_revoluteJointDef, &_revoluteJointDef.motorSpeed);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		ypos = createJointInputGui(_toolOptions, ypos);
	}

	void WeldTool::initGui()
	{
		// Create GUI
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		// Collide with connected
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Collide connected");

		BoolOption<b2WeldJointDef>* bo = 
			new BoolOption<b2WeldJointDef>(_toolOptions,
			&_weldJointDef, &_weldJointDef.collideConnected);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Reference angle
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Reference angle");

		FloatOption<b2WeldJointDef>* fo = 
			new FloatOption<b2WeldJointDef>(_toolOptions,
			&_weldJointDef, &_weldJointDef.referenceAngle);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Frequency
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Frequency");

		fo = 
			new FloatOption<b2WeldJointDef>(_toolOptions,
			&_weldJointDef, &_weldJointDef.frequencyHz);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Damping ratio
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Damping ratio");
		
		SliderOption<b2WeldJointDef>* so =
			new SliderOption<b2WeldJointDef>(_toolOptions,
			&_weldJointDef, &_weldJointDef.dampingRatio);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT;
	}

	void WheelTool::initGui()
	{
		// Create GUI
		const float LINE_HEIGHT = 22;

		const float xstart = 100;
		const float xwidth = 220;

		float ypos = 0;

		// Collide with connected
		Gwen::Controls::Label* label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Collide connected");

		BoolOption<b2WheelJointDef>* bo = 
			new BoolOption<b2WheelJointDef>(_toolOptions,
			&_wheelJointDef, &_wheelJointDef.collideConnected);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Enable motor
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Enable motor");

		bo = 
			new BoolOption<b2WheelJointDef>(_toolOptions,
			&_wheelJointDef, &_wheelJointDef.enableMotor);
		bo->getCheckBox()->SetPos(xstart, ypos);
		_updatableOptions.push_back(bo);

		ypos += LINE_HEIGHT;

		// Max force
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Max motor torque");

		FloatOption<b2WheelJointDef>* fo = 
			new FloatOption<b2WheelJointDef>(_toolOptions,
			&_wheelJointDef, &_wheelJointDef.maxMotorTorque);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Motor speed
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("  Motor speed");

		fo = 
			new FloatOption<b2WheelJointDef>(_toolOptions,
			&_wheelJointDef, &_wheelJointDef.motorSpeed);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);

		ypos += LINE_HEIGHT;

		// Frequency
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Frequency");

		fo = 
			new FloatOption<b2WheelJointDef>(_toolOptions,
			&_wheelJointDef, &_wheelJointDef.frequencyHz);
		fo->getTextBox()->SetPos(xstart, ypos);
		fo->getTextBox()->SetWidth(xwidth - xstart);
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Damping ratio
		label = new Gwen::Controls::Label(_toolOptions);
		label->SetPos(5, ypos);
		label->SetText("Damping ratio");
		
		SliderOption<b2WheelJointDef>* so =
			new SliderOption<b2WheelJointDef>(_toolOptions,
			&_wheelJointDef, &_wheelJointDef.dampingRatio);
		so->getSlider()->SetPos(xstart, ypos);
		so->getSlider()->SetWidth(xwidth - xstart);
		so->getSlider()->SetRange(0.0f, 1.0f);
		so->getSlider()->SetHeight(15);
		_updatableOptions.push_back(so);

		ypos += LINE_HEIGHT;

		ypos = createJointInputGui(_toolOptions, ypos);
	}
}
