#include "tools/Tool.h"

#include <Box2D/Box2D.h>

#include "tools/tools/SelectionTool.h"

#include "tools/joints/DistanceTool.h"
#include "tools/joints/PrismaticTool.h"
#include "tools/joints/PulleyTool.h"
#include "tools/joints/RevoluteTool.h"
#include "tools/joints/WeldTool.h"
#include "tools/joints/WheelTool.h"

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
		// Body options
		Tool::createLabel("Body options (new body)", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		// Body type
		MultiOption<b2BodyDef, b2BodyType>* mo =
			createMultiOption("  Body type", parent, 5, ypos,
			GetterSetter<b2BodyDef, b2BodyType>(&_bodyDef, &b2BodyDef::type));
		mo->addOption("Static", b2_staticBody);
		mo->addOption("Dynamic", b2_dynamicBody);
		_updatableOptions.push_back(mo);
		ypos += LINE_HEIGHT;
		
		// Active
		BoolOption<b2BodyDef>* bo = Tool::createBoolOption("  Active",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, bool>(&_bodyDef, &b2BodyDef::active));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Allow sleep
		bo = Tool::createBoolOption("  Allow sleep",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, bool>(&_bodyDef, &b2BodyDef::allowSleep));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Awake
		bo = Tool::createBoolOption("  Awake",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, bool>(&_bodyDef, &b2BodyDef::awake));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Fix rotation
		bo = Tool::createBoolOption("  Fix rotation",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, bool>(&_bodyDef, &b2BodyDef::fixedRotation));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Is fast moving
		bo = Tool::createBoolOption("  Is fast moving",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, bool>(&_bodyDef, &b2BodyDef::bullet));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Gravity scale
		FloatOption<b2BodyDef>* fo = Tool::createFloatOption("  Gravity scale",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, float>(&_bodyDef, &b2BodyDef::gravityScale));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT*2;

		// Initial velocities
		Tool::createLabel("  Initial velocities", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		// Linear
		VectorOption<b2BodyDef>* vo = Tool::createVectorOption("    Linear",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, b2Vec2>(&_bodyDef, &b2BodyDef::linearVelocity));
		_updatableOptions.push_back(vo);
		ypos += LINE_HEIGHT * 2;
		
		// Angular
		fo = Tool::createFloatOption("    Angular",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, float>(&_bodyDef, &b2BodyDef::angularVelocity));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT*2;

		// Velocity damping
		Tool::createLabel("  Velocity damping", parent, 5, ypos);
		ypos += LINE_HEIGHT;
		
		// Linear
		SliderOption<b2BodyDef>* so = Tool::createSliderOption("    Linear",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, float>(&_bodyDef, &b2BodyDef::linearDamping),
			0.0f, 1.0f);
		_updatableOptions.push_back(so);
		ypos += LINE_HEIGHT;
		
		// Angular
		so = Tool::createSliderOption("    Angular",
			parent, 5, ypos,
			GetterSetter<b2BodyDef, float>(&_bodyDef, &b2BodyDef::angularDamping),
			0.0f, 1.0f);
		_updatableOptions.push_back(so);
		ypos += LINE_HEIGHT*2;

		return ypos;
	}
	
	float Tool::createFixtureGui(Gwen::Controls::Base* parent, float ypos)
	{
		// Fixture options
		Tool::createLabel("Fixture options", parent, 5, ypos);
		ypos += LINE_HEIGHT;
		
		// Friction
		SliderOption<b2FixtureDef>* so =
			createSliderOption("    Friction", parent, 5, ypos,
			GetterSetter<b2FixtureDef, float>(&_fixtureDef, &b2FixtureDef::friction), 0.0f, 1.0f);
		_updatableOptions.push_back(so);
		ypos += LINE_HEIGHT;
		
		// Restitution
		so =
			createSliderOption("    Restitution", parent, 5, ypos,
			GetterSetter<b2FixtureDef, float>(&_fixtureDef, &b2FixtureDef::restitution), 0.0f, 1.0f);
		_updatableOptions.push_back(so);
		ypos += LINE_HEIGHT;
		
		// Density
		FloatOption<b2FixtureDef>* fo =
			createFloatOption("    Density", parent, 5, ypos,
			GetterSetter<b2FixtureDef, float>(&_fixtureDef, &b2FixtureDef::density));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT * 2;

		return ypos;
	}

	float Tool::createJointInputGui(Gwen::Controls::Base* parent, float ypos)
	{
		// Input options
		Tool::createLabel("Input options", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		// Enabled
		BoolOption<MotorInput>* bo =
			createBoolOption("  Enabled", parent, 5, ypos,
			GetterSetter<MotorInput, bool>(&_motorInput,
			&MotorInput::getEnabled, &MotorInput::setEnabled));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Max force
		FloatOption<MotorInput>* fo =
			createFloatOption("  Max force", parent, 5, ypos,
			GetterSetter<MotorInput, float>(&_motorInput,
			&MotorInput::getMaxForce, &MotorInput::setMaxForce));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Speed
		fo =
			createFloatOption("  Speed", parent, 5, ypos,
			GetterSetter<MotorInput, float>(&_motorInput,
			&MotorInput::getSpeed, &MotorInput::setSpeed));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Forward key
		Tool::createLabel("  Forward key", parent, 5, ypos);

		Gwen::Controls::Button* _forwardButton = new Button(parent);
		_forwardButton->SetPos(OPTIONS_X_START, ypos);
		_forwardButton->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);
		_forwardButton->SetText(InputConverter::translateCharToString(_motorInput.getForwardKey()));
		_forwardButton->UserData.Set("updateKey", &_motorInput._forwardButton);
		_forwardButton->SetIsToggle(true);

		_forwardButtons.push_back(_forwardButton);

		ypos += LINE_HEIGHT;

		// Reverse key
		Tool::createLabel("  Reverse key", parent, 5, ypos);

		Gwen::Controls::Button* _reverseButton = new Button(parent);
		_reverseButton->SetPos(OPTIONS_X_START, ypos);
		_reverseButton->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);
		_reverseButton->SetText(InputConverter::translateCharToString(_motorInput.getReverseKey()));
		_reverseButton->UserData.Set("updateKey", &_motorInput._reverseButton);
		_reverseButton->SetIsToggle(true);

		_reverseButtons.push_back(_reverseButton);

		ypos += LINE_HEIGHT;

		return ypos;
	}

	float SelectionTool::createJointInputGuiSelection(
		Gwen::Controls::Base* parent, float ypos, b2Joint* joint)
	{
		MotorInput* motorInput = (MotorInput*)joint->GetUserData();

		if (motorInput != nullptr)
		{
			// Input options
			Tool::createLabel("Input options", parent, 5, ypos);
			ypos += LINE_HEIGHT;

			// Enabled
			BoolOption<MotorInput>* bo =
				createBoolOption("  Enabled", parent, 5, ypos,
				GetterSetter<MotorInput, bool>(motorInput, &MotorInput::getEnabled, &MotorInput::setEnabled));
			_jointOptionsUpdatables.push_back(bo);
			ypos += LINE_HEIGHT;

			// Max force
			FloatOption<MotorInput>* fo =
				createFloatOption("  Max force", parent, 5, ypos,
				GetterSetter<MotorInput, float>(motorInput, &MotorInput::getMaxForce, &MotorInput::setMaxForce));
			_jointOptionsUpdatables.push_back(fo);
			ypos += LINE_HEIGHT;

			// Speed
			fo =
				createFloatOption("  Speed", parent, 5, ypos,
				GetterSetter<MotorInput, float>(motorInput, &MotorInput::getSpeed, &MotorInput::setSpeed));
			_jointOptionsUpdatables.push_back(fo);
			ypos += LINE_HEIGHT;

			// Forward key
			Tool::createLabel("  Forward key", parent, 5, ypos);

			Gwen::Controls::Button* _forwardButton = new Button(parent);
			_forwardButton->SetPos(OPTIONS_X_START, ypos);
			_forwardButton->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);
			_forwardButton->SetText(InputConverter::translateCharToString(motorInput->getForwardKey()));
			_forwardButton->UserData.Set("updateKey", &motorInput->_forwardButton);
			_forwardButton->SetIsToggle(true);

			_inputButtons.push_back(_forwardButton);

			ypos += LINE_HEIGHT;

			Tool::createLabel("  Reverse key", parent, 5, ypos);

			Gwen::Controls::Button* _reverseButton = new Button(parent);
			_reverseButton->SetPos(OPTIONS_X_START, ypos);
			_reverseButton->SetWidth(OPTIONS_X_WIDTH - OPTIONS_X_START);
			_reverseButton->SetText(InputConverter::translateCharToString(motorInput->getReverseKey()));
			_reverseButton->UserData.Set("updateKey", &motorInput->_reverseButton);
			_reverseButton->SetIsToggle(true);

			_inputButtons.push_back(_reverseButton);
		}

		return ypos;
	}

	void SelectionTool::initGui()
	{
		float ypos = 0;

		Gwen::Controls::Base* parent = _otherOptions;

		// Body options
		Tool::createLabel("Body options (existing body)", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		// Body type
		MultiOption<b2Body, b2BodyType>* mo =
			createMultiOption("  Body type", parent, 5, ypos,
			GetterSetter<b2Body, b2BodyType>(nullptr, &b2Body::GetType, &b2Body::SetType));
		mo->addOption("Static", b2_staticBody);
		mo->addOption("Dynamic", b2_dynamicBody);

		_updatableOptions.push_back(mo);
		_bodyOptions.push_back(mo);
		ypos += LINE_HEIGHT;
		
		// Active
		BoolOption<b2Body>* bo =
			createBoolOption("  Active", parent, 5, ypos,
			GetterSetter<b2Body, bool>(nullptr, &b2Body::IsActive, &b2Body::SetActive));

		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Allow sleep
		bo =
			createBoolOption("  Allow sleep", parent, 5, ypos,
			GetterSetter<b2Body, bool>(nullptr, &b2Body::IsSleepingAllowed, &b2Body::SetSleepingAllowed));

		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Awake
		bo =
			createBoolOption("  Awake", parent, 5, ypos,
			GetterSetter<b2Body, bool>(nullptr, &b2Body::IsAwake, &b2Body::SetAwake));

		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Fix rotation
		bo =
			createBoolOption("  Fix rotation", parent, 5, ypos,
			GetterSetter<b2Body, bool>(nullptr, &b2Body::IsFixedRotation, &b2Body::SetFixedRotation));

		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Is fast moving
		bo =
			createBoolOption("  Is fast moving", parent, 5, ypos,
			GetterSetter<b2Body, bool>(nullptr, &b2Body::IsBullet, &b2Body::SetBullet));

		_updatableOptions.push_back(bo);
		_bodyOptions.push_back(bo);
		ypos += LINE_HEIGHT;
		
		// Gravity Scale
		FloatOption<b2Body>* fo =
			createFloatOption("  Gravity scale", parent, 5, ypos,
			GetterSetter<b2Body, float>(nullptr, &b2Body::GetGravityScale, &b2Body::SetGravityScale));

		_updatableOptions.push_back(fo);
		_bodyOptions.push_back(fo);
		ypos += LINE_HEIGHT*2;

		// Initial velocities
		Tool::createLabel("  Initial velocities", parent, 5, ypos);
		ypos += LINE_HEIGHT;
		
		// Linear
		VectorOption<b2Body>* vo =
			createVectorOption("    Linear", parent, 5, ypos,
			GetterSetter<b2Body, b2Vec2>(nullptr, &b2Body::GetLinearVelocity, &b2Body::SetLinearVelocity));

		_updatableOptions.push_back(vo);
		_bodyOptions.push_back(vo);
		ypos += LINE_HEIGHT*2;

		// Angular
		fo = createFloatOption("    Angular", parent, 5, ypos,
		GetterSetter<b2Body, float>(nullptr, &b2Body::GetAngularVelocity, &b2Body::SetAngularVelocity));

		_updatableOptions.push_back(fo);
		_bodyOptions.push_back(fo);
		ypos += LINE_HEIGHT*2;

		// Velocity Damping
		Tool::createLabel("  Velocity damping", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		// Linear
		SliderOption<b2Body>* so =
			createSliderOption("    Linear", parent, 5, ypos,
			GetterSetter<b2Body, float>(nullptr, &b2Body::GetLinearDamping,
			&b2Body::SetLinearDamping), 0.0f, 1.0f);

		_updatableOptions.push_back(so);
		_bodyOptions.push_back(so);
		ypos += LINE_HEIGHT;

		// Angular
		so =
			createSliderOption("    Angular", parent, 5, ypos,
			GetterSetter<b2Body, float>(nullptr, &b2Body::GetAngularDamping,
			&b2Body::SetAngularDamping), 0.0f, 1.0f);

		_updatableOptions.push_back(so);
		_bodyOptions.push_back(so);
		ypos += LINE_HEIGHT*2;

		// Fixture options
		Tool::createLabel("Fixture options", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		// Friction
		SliderOption<b2Fixture>* sof =
			createSliderOption("    Friction", parent, 5, ypos,
			GetterSetter<b2Fixture, float>(nullptr, &b2Fixture::GetFriction,
			&b2Fixture::SetFriction), 0.0f, 1.0f);

		_updatableOptions.push_back(sof);
		_fixtureOptions.push_back(sof);
		ypos += LINE_HEIGHT;

		// Restitution
		sof =
			createSliderOption("    Restitution", parent, 5, ypos,
			GetterSetter<b2Fixture, float>(nullptr, &b2Fixture::GetRestitution,
			&b2Fixture::SetRestitution), 0.0f, 1.0f);

		_updatableOptions.push_back(sof);
		_fixtureOptions.push_back(sof);
		ypos += LINE_HEIGHT;

		// Density
		FloatOption<b2Fixture>* fof =
			createFloatOption("    Density", parent, 5, ypos,
			GetterSetter<b2Fixture, float>(nullptr, &b2Fixture::GetDensity, &b2Fixture::SetDensity));

		_updatableOptions.push_back(fof);
		_fixtureOptions.push_back(fof);
		ypos += LINE_HEIGHT;
	}

	void SelectionTool::selectJoint(b2Joint* joint)
	{
		float ypos = 10;

		_inputButtons.clear();

		delete _jointOptions;
		_jointOptions = new Base(_jointOptionsPanel);
		_jointOptions->Dock(Pos::Fill);

		auto parent = _jointOptions;

		// Joint options
		Tool::createLabel("Joint options", parent, 5, ypos);
		ypos += LINE_HEIGHT;

		switch (joint->GetType())
		{
		case b2JointType::e_distanceJoint:
			{
				b2DistanceJoint* specJoint = (b2DistanceJoint*)joint;

				// Collide connected
				BoolOption<b2DistanceJoint>* bo =
					createBoolOption("Collide connected", parent, 5, ypos,
					GetterSetter<b2DistanceJoint, bool>(specJoint,
					&b2DistanceJoint::GetCollideConnected));
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Length
				FloatOption<b2DistanceJoint>* fo =
					createFloatOption("Length", parent, 5, ypos,
					GetterSetter<b2DistanceJoint, float>(specJoint,
					&b2DistanceJoint::GetLength, &b2DistanceJoint::SetLength));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Frequency
				fo =
					createFloatOption("Frequency", parent, 5, ypos,
					GetterSetter<b2DistanceJoint, float>(specJoint,
					&b2DistanceJoint::GetFrequency, &b2DistanceJoint::SetFrequency));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Damping ratio
				SliderOption<b2DistanceJoint>* so =
					createSliderOption("Damping ratio", parent, 5, ypos,
					GetterSetter<b2DistanceJoint, float>(specJoint,
					&b2DistanceJoint::GetDampingRatio,
					&b2DistanceJoint::SetDampingRatio), 0.0f, 1.0f);
				_jointOptionsUpdatables.push_back(so);
				ypos += LINE_HEIGHT;
			}
			break;
		case b2JointType::e_prismaticJoint:
			{
				b2PrismaticJoint* specJoint = (b2PrismaticJoint*)joint;

				// Collide connected
				BoolOption<b2PrismaticJoint>* bo =
					createBoolOption("Collide connected", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, bool>(specJoint,
					&b2PrismaticJoint::GetCollideConnected));
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Reference angle
				FloatOption<b2PrismaticJoint>* fo =
					createFloatOption("Reference angle", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, float>(specJoint,
					&b2PrismaticJoint::GetReferenceAngle));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Enable limits
				bo =
					createBoolOption("Enable limits", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, bool>(specJoint,
					&b2PrismaticJoint::IsLimitEnabled, &b2PrismaticJoint::EnableLimit));
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Lower limit
				fo =
					createFloatOption("Lower limit", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, float>(specJoint,
					&b2PrismaticJoint::GetLowerLimit));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Upper limit
				fo =
					createFloatOption("Upper limit", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, float>(specJoint,
					&b2PrismaticJoint::GetUpperLimit));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Enable motor
				bo =
					createBoolOption("Enable motor", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, bool>(specJoint,
					&b2PrismaticJoint::IsMotorEnabled, &b2PrismaticJoint::EnableMotor));
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Max force
				fo =
					createFloatOption("Max force", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, float>(specJoint,
					&b2PrismaticJoint::GetMaxMotorForce, &b2PrismaticJoint::SetMaxMotorForce));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Motor speed
				fo =
					createFloatOption("Motor speed", parent, 5, ypos,
					GetterSetter<b2PrismaticJoint, float>(specJoint,
					&b2PrismaticJoint::GetMotorSpeed, &b2PrismaticJoint::SetMotorSpeed));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;
			}
			break;
		case b2JointType::e_pulleyJoint:
			{
				b2PulleyJoint* specJoint = (b2PulleyJoint*)joint;

				// Collide connected
				BoolOption<b2PulleyJoint>* bo =
					createBoolOption("Collide connected", parent, 5, ypos,
					GetterSetter<b2PulleyJoint, bool>(specJoint,
					&b2PulleyJoint::GetCollideConnected));
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Length A
				FloatOption<b2PulleyJoint>* fo =
					createFloatOption("Length A", parent, 5, ypos,
					GetterSetter<b2PulleyJoint, float>(specJoint,
					&b2PulleyJoint::GetLengthA));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Length B
				fo =
					createFloatOption("Length B", parent, 5, ypos,
					GetterSetter<b2PulleyJoint, float>(specJoint,
					&b2PulleyJoint::GetLengthA));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Length B
				fo =
					createFloatOption("Ratio", parent, 5, ypos,
					GetterSetter<b2PulleyJoint, float>(specJoint,
					&b2PulleyJoint::GetRatio));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;
			}
			break;
		case b2JointType::e_revoluteJoint:
			{
				b2RevoluteJoint* specJoint = (b2RevoluteJoint*)joint;

				// Collide connected
				BoolOption<b2RevoluteJoint>* bo =
					createBoolOption("Collide connected", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, bool>(specJoint,
					&b2RevoluteJoint::GetCollideConnected));
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Reference angle
				FloatOption<b2RevoluteJoint>* fo =
					createFloatOption("Reference angle", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, float>(specJoint,
					&b2RevoluteJoint::GetReferenceAngle));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Enable limits
				bo =
					createBoolOption("Enable limits", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, bool>(specJoint,
					&b2RevoluteJoint::IsLimitEnabled, &b2RevoluteJoint::EnableLimit));
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Lower limit
				fo =
					createFloatOption("  Lower angle", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, float>(specJoint,
					&b2RevoluteJoint::GetLowerLimit));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Upper limit
				fo =
					createFloatOption("  Upper angle", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, float>(specJoint,
					&b2RevoluteJoint::GetUpperLimit));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Enable motor
				bo =
					createBoolOption("Enable motor", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, bool>(specJoint,
					&b2RevoluteJoint::IsMotorEnabled, &b2RevoluteJoint::EnableMotor));
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Max force
				fo =
					createFloatOption("  Max force", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, float>(specJoint,
					&b2RevoluteJoint::GetMaxMotorTorque, &b2RevoluteJoint::SetMaxMotorTorque));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Motor speed
				fo =
					createFloatOption("  Motor speed", parent, 5, ypos,
					GetterSetter<b2RevoluteJoint, float>(specJoint,
					&b2RevoluteJoint::GetMotorSpeed, &b2RevoluteJoint::SetMotorSpeed));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;
			}
			break;
		case b2JointType::e_weldJoint:
			{
				b2WeldJoint* specJoint = (b2WeldJoint*)joint;

				// Collide connected
				BoolOption<b2WeldJoint>* bo =
					createBoolOption("Collide connected", parent, 5, ypos,
					GetterSetter<b2WeldJoint, bool>(specJoint,
					&b2WeldJoint::GetCollideConnected));
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Reference angle
				FloatOption<b2WeldJoint>* fo =
					createFloatOption("Reference angle", parent, 5, ypos,
					GetterSetter<b2WeldJoint, float>(specJoint,
					&b2WeldJoint::GetReferenceAngle));
				fo->getTextBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Frequency
				fo =
					createFloatOption("Frequency", parent, 5, ypos,
					GetterSetter<b2WeldJoint, float>(specJoint,
					&b2WeldJoint::GetFrequency, &b2WeldJoint::SetFrequency));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Damping ratio
				SliderOption<b2WeldJoint>* so =
					createSliderOption("Damping ratio", parent, 5, ypos,
					GetterSetter<b2WeldJoint, float>(specJoint,
					&b2WeldJoint::GetDampingRatio,
					&b2WeldJoint::SetDampingRatio), 0.0f, 1.0f);
				_jointOptionsUpdatables.push_back(so);
				ypos += LINE_HEIGHT;
			}
			break;
		case b2JointType::e_wheelJoint:
			{
				b2WheelJoint* specJoint = (b2WheelJoint*)joint;

				// Collide connected
				BoolOption<b2WheelJoint>* bo =
					createBoolOption("Collide connected", parent, 5, ypos,
					GetterSetter<b2WheelJoint, bool>(specJoint,
					&b2WheelJoint::GetCollideConnected));
				bo->getCheckBox()->SetDisabled(true);
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Enable motor
				bo =
					createBoolOption("Enable motor", parent, 5, ypos,
					GetterSetter<b2WheelJoint, bool>(specJoint,
					&b2WheelJoint::IsMotorEnabled, &b2WheelJoint::EnableMotor));
				_jointOptionsUpdatables.push_back(bo);
				ypos += LINE_HEIGHT;

				// Max force
				FloatOption<b2WheelJoint>* fo =
					createFloatOption("  Max force", parent, 5, ypos,
					GetterSetter<b2WheelJoint, float>(specJoint,
					&b2WheelJoint::GetMaxMotorTorque, &b2WheelJoint::SetMaxMotorTorque));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Motor speed
				fo =
					createFloatOption("  Motor speed", parent, 5, ypos,
					GetterSetter<b2WheelJoint, float>(specJoint,
					&b2WheelJoint::GetMotorSpeed, &b2WheelJoint::SetMotorSpeed));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Frequency
				fo =
					createFloatOption("Frequency", parent, 5, ypos,
					GetterSetter<b2WheelJoint, float>(specJoint,
					&b2WheelJoint::GetSpringFrequencyHz, &b2WheelJoint::SetSpringFrequencyHz));
				_jointOptionsUpdatables.push_back(fo);
				ypos += LINE_HEIGHT;

				// Damping ratio
				SliderOption<b2WheelJoint>* so =
					createSliderOption("Damping ratio", parent, 5, ypos,
					GetterSetter<b2WheelJoint, float>(specJoint,
					&b2WheelJoint::GetSpringDampingRatio,
					&b2WheelJoint::SetSpringDampingRatio), 0.0f, 1.0f);
				_jointOptionsUpdatables.push_back(so);
				ypos += LINE_HEIGHT;
			}
			break;
		}

		ypos = createJointInputGuiSelection(_jointOptions, ypos, joint);

		ypos += LINE_HEIGHT;

		auto button = new Gwen::Controls::Button(_jointOptions);
		button->SetPos(5, ypos);
		button->SetText("Delete Joint");
		button->onPress.Add(this, &SelectionTool::deleteJoint);
	}

	void DistanceTool::initGui()
	{
		float ypos = 0;
		auto parent = _toolOptions;

		// Collide connected
		BoolOption<b2DistanceJointDef>* bo =
			createBoolOption("Collide connected", parent, 5, ypos,
			GetterSetter<b2DistanceJointDef, bool>(&_distanceJointDef,
			&b2DistanceJointDef::collideConnected));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Length
		FloatOption<b2DistanceJointDef>* fo =
			createFloatOption("Length", parent, 5, ypos,
			GetterSetter<b2DistanceJointDef, float>(&_distanceJointDef,
			&b2DistanceJointDef::length));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Frequency
		fo =
			createFloatOption("Frequency", parent, 5, ypos,
			GetterSetter<b2DistanceJointDef, float>(&_distanceJointDef,
			&b2DistanceJointDef::frequencyHz));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Damping ratio
		SliderOption<b2DistanceJointDef>* so =
			createSliderOption("Damping ratio", parent, 5, ypos,
			GetterSetter<b2DistanceJointDef, float>(&_distanceJointDef,
			&b2DistanceJointDef::dampingRatio), 0.0f, 1.0f);
		_updatableOptions.push_back(so);
		ypos += LINE_HEIGHT;
	}

	void PrismaticTool::initGui()
	{
		float ypos = 0;
		auto parent = _toolOptions;

		// Collide connected
		BoolOption<b2PrismaticJointDef>* bo =
			createBoolOption("Collide connected", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, bool>(&_prismaticJointDef,
			&b2PrismaticJointDef::collideConnected));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Reference angle
		FloatOption<b2PrismaticJointDef>* fo =
			createFloatOption("Reference angle", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, float>(&_prismaticJointDef,
			&b2PrismaticJointDef::referenceAngle));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Enable limits
		bo =
			createBoolOption("Enable limits", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, bool>(&_prismaticJointDef,
			&b2PrismaticJointDef::enableLimit));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Lower limit
		fo =
			createFloatOption("Lower limit", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, float>(&_prismaticJointDef,
			&b2PrismaticJointDef::lowerTranslation));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Upper limit
		fo =
			createFloatOption("Upper limit", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, float>(&_prismaticJointDef,
			&b2PrismaticJointDef::upperTranslation));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Enable motor
		bo =
			createBoolOption("Enable motor", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, bool>(&_prismaticJointDef,
			&b2PrismaticJointDef::enableMotor));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Max force
		fo =
			createFloatOption("Max force", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, float>(&_prismaticJointDef,
			&b2PrismaticJointDef::maxMotorForce));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Motor speed
		fo =
			createFloatOption("Motor speed", parent, 5, ypos,
			GetterSetter<b2PrismaticJointDef, float>(&_prismaticJointDef,
			&b2PrismaticJointDef::motorSpeed));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Create input options
		ypos = createJointInputGui(parent, ypos);
	}

	void PulleyTool::initGui()
	{
		float ypos = 0;
		auto parent = _toolOptions;

		// Collide connected
		BoolOption<b2PulleyJointDef>* bo =
			createBoolOption("Collide connected", parent, 5, ypos,
			GetterSetter<b2PulleyJointDef, bool>(&_pulleyJointDef,
			&b2PulleyJointDef::collideConnected));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Length A
		FloatOption<b2PulleyJointDef>* fo =
			createFloatOption("Length A", parent, 5, ypos,
			GetterSetter<b2PulleyJointDef, float>(&_pulleyJointDef,
			&b2PulleyJointDef::lengthA));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Length B
		fo =
			createFloatOption("Length B", parent, 5, ypos,
			GetterSetter<b2PulleyJointDef, float>(&_pulleyJointDef,
			&b2PulleyJointDef::lengthB));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Length B
		fo =
			createFloatOption("Ratio", parent, 5, ypos,
			GetterSetter<b2PulleyJointDef, float>(&_pulleyJointDef,
			&b2PulleyJointDef::ratio));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;
	}

	void RevoluteTool::initGui()
	{
		float ypos = 0;
		auto parent = _toolOptions;

		// Collide connected
		BoolOption<b2RevoluteJointDef>* bo =
			createBoolOption("Collide connected", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, bool>(&_revoluteJointDef,
			&b2RevoluteJointDef::collideConnected));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Reference angle
		FloatOption<b2RevoluteJointDef>* fo =
			createFloatOption("Reference angle", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, float>(&_revoluteJointDef,
			&b2RevoluteJointDef::referenceAngle));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Enable limits
		bo =
			createBoolOption("Enable limits", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, bool>(&_revoluteJointDef,
			&b2RevoluteJointDef::enableLimit));
		_updatableOptions.push_back(bo);
		ypos += LINE_HEIGHT;

		// Lower limit
		fo =
			createFloatOption("  Lower angle", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, float>(&_revoluteJointDef,
			&b2RevoluteJointDef::lowerAngle));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Upper limit
		fo =
			createFloatOption("  Upper angle", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, float>(&_revoluteJointDef,
			&b2RevoluteJointDef::upperAngle));
		_updatableOptions.push_back(fo);
		ypos += LINE_HEIGHT;

		// Enable motor
		bo =
			createBoolOption("Enable motor", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, bool>(&_revoluteJointDef,
			&b2RevoluteJointDef::enableMotor));
		_jointOptionsUpdatables.push_back(bo);
		ypos += LINE_HEIGHT;

		// Max force
		fo =
			createFloatOption("  Max force", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, float>(&_revoluteJointDef,
			&b2RevoluteJointDef::maxMotorTorque));
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Motor speed
		fo =
			createFloatOption("  Motor speed", parent, 5, ypos,
			GetterSetter<b2RevoluteJointDef, float>(&_revoluteJointDef,
			&b2RevoluteJointDef::motorSpeed));
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Create input options
		ypos = createJointInputGui(parent, ypos);
	}

	void WeldTool::initGui()
	{
		float ypos = 0;
		auto parent = _toolOptions;

		// Collide connected
		BoolOption<b2WeldJointDef>* bo =
			createBoolOption("Collide connected", parent, 5, ypos,
			GetterSetter<b2WeldJointDef, bool>(&_weldJointDef,
			&b2WeldJointDef::collideConnected));
		bo->getCheckBox()->SetDisabled(true);
		_jointOptionsUpdatables.push_back(bo);
		ypos += LINE_HEIGHT;

		// Reference angle
		FloatOption<b2WeldJointDef>* fo =
			createFloatOption("Reference angle", parent, 5, ypos,
			GetterSetter<b2WeldJointDef, float>(&_weldJointDef,
			&b2WeldJointDef::referenceAngle));
		fo->getTextBox()->SetDisabled(true);
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Frequency
		fo =
			createFloatOption("Frequency", parent, 5, ypos,
			GetterSetter<b2WeldJointDef, float>(&_weldJointDef,
			&b2WeldJointDef::frequencyHz));
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Damping ratio
		SliderOption<b2WeldJointDef>* so =
			createSliderOption("Damping ratio", parent, 5, ypos,
			GetterSetter<b2WeldJointDef, float>(&_weldJointDef,
			&b2WeldJointDef::dampingRatio), 0.0f, 1.0f);
		_jointOptionsUpdatables.push_back(so);
		ypos += LINE_HEIGHT;
	}

	void WheelTool::initGui()
	{
		float ypos = 0;
		auto parent = _toolOptions;

		// Collide connected
		BoolOption<b2WheelJointDef>* bo =
			createBoolOption("Collide connected", parent, 5, ypos,
			GetterSetter<b2WheelJointDef, bool>(&_wheelJointDef,
			&b2WheelJointDef::collideConnected));
		bo->getCheckBox()->SetDisabled(true);
		_jointOptionsUpdatables.push_back(bo);
		ypos += LINE_HEIGHT;

		// Enable motor
		bo =
			createBoolOption("Enable motor", parent, 5, ypos,
			GetterSetter<b2WheelJointDef, bool>(&_wheelJointDef,
			&b2WheelJointDef::enableMotor));
		_jointOptionsUpdatables.push_back(bo);
		ypos += LINE_HEIGHT;

		// Max force
		FloatOption<b2WheelJointDef>* fo =
			createFloatOption("  Max force", parent, 5, ypos,
			GetterSetter<b2WheelJointDef, float>(&_wheelJointDef,
			&b2WheelJointDef::maxMotorTorque));
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Motor speed
		fo =
			createFloatOption("  Motor speed", parent, 5, ypos,
			GetterSetter<b2WheelJointDef, float>(&_wheelJointDef,
			&b2WheelJointDef::motorSpeed));
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Frequency
		fo =
			createFloatOption("Frequency", parent, 5, ypos,
			GetterSetter<b2WheelJointDef, float>(&_wheelJointDef,
			&b2WheelJointDef::frequencyHz));
		_jointOptionsUpdatables.push_back(fo);
		ypos += LINE_HEIGHT;

		// Damping ratio
		SliderOption<b2WheelJointDef>* so =
			createSliderOption("Damping ratio", parent, 5, ypos,
			GetterSetter<b2WheelJointDef, float>(&_wheelJointDef,
			&b2WheelJointDef::dampingRatio), 0.0f, 1.0f);
		_jointOptionsUpdatables.push_back(so);
		ypos += LINE_HEIGHT;

		// Create input options
		ypos = createJointInputGui(parent, ypos);
	}
}
