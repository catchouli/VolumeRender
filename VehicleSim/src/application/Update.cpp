#include "VehicleSim.h"

#include "serialisation/Serialiser.h"

namespace vlr
{
	const char* VEHICLESIM_TITLE_FORMAT = "Physical sandbox [FPS: %d]\n";

	void VehicleSim::doStep()
	{
		_physWorld.Step(1.0f / _timeStep, _velocityIterations, _positionIterations);
	}

	void VehicleSim::update(double dt)
	{
		// Update options
		if (!_worldOptionsTabButton->Hidden())
		{
			for (auto it = _updatableOptions.begin(); it != _updatableOptions.end(); ++it)
			{
				(*it)->update();
			}
		}

		// Do frame update
		doFrameInput(dt);

		// Get inner area of dock
		Gwen::Rect innerBounds = _guiDock->GetInnerBounds();

		// Update camera's viewport
		_camera.setViewport(innerBounds.x, innerBounds.y, innerBounds.w, innerBounds.h);

		// Update camera's matrix
		if (innerBounds.h > 0)
		{
			float aspect = (float)innerBounds.w / innerBounds.h;
			_camera.orthographic(_orthoScale, aspect);
		}

		// Handle motor for each joint
		for (auto joint = _physWorld.GetJointList(); joint; joint = joint->GetNext())
		{
			MotorInput* motorInput = (MotorInput*)joint->GetUserData();

			if (motorInput != nullptr)
			{
				motorInput->update(_window, joint);
			}
		}

		// Update physics system
		double time = glfwGetTime();
		float stepTime = (1.0f / _timeStep);
		while (_lastPhysicsUpdate + stepTime < time)
		{
			if (_simulationRunning)
				doStep();

			_lastPhysicsUpdate += stepTime;
		}

		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, VEHICLESIM_TITLE_FORMAT, getFPS());
		glfwSetWindowTitle(_window, title);
	}
}