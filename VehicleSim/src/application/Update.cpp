#include "VehicleSim.h"

#include "serialisation/Serialiser.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace vlr
{
	const char* VEHICLESIM_TITLE_FORMAT = "Physical sandbox [FPS: %d]\n";

	void VehicleSim::doStep()
	{
		_physWorld.Step(1.0f / _timeStep, _velocityIterations, _positionIterations);
	}

	float getAngle(const b2Vec2& a, const b2Vec2& b)
	{
		float dot = (a.x * b.x + a.y * b.y);
		float det = (a.x * b.y - a.y * b.x);

		return atan2f(det, dot);
	}

	void VehicleSim::update(double dt)
	{
		for (b2Body* body = _physWorld.GetBodyList(); body; body = body->GetNext())
		{
			//body->ApplyForceToCenter(body->getForces(), true);
			//return;
			if (!body->IsActive())
				continue;

			// Simulate rolling resistance
			// Simple version of rolling resistance using formula given by spec?
			// Frr = -Crr * v
			// However, wouldn't rolling resistance counteract the force due to gravity
			// such that a coefficient of 1 would equal no movement?
			// To calculate this, you could resolve the gravity into the direction of
			// the vector perpendicular to the normal of the contact and then multiply that
			// by the coefficient of friction of the contact
			for (b2ContactEdge* contactEdge = body->GetContactList(); contactEdge; contactEdge = contactEdge->next)
			{
				b2Contact* contact = contactEdge->contact;

				// Get fixtures
				b2Fixture* fixA = contact->GetFixtureA();
				b2Fixture* fixB = contact->GetFixtureB();
				
				b2Fixture* bodyFix = (fixA->GetBody() == body ? fixA : fixB);

				if (bodyFix->GetShape()->GetType() == b2Shape::e_circle)
				{
					// Get friction of fixtures
					float ca = fixA->GetFriction();
					float cb = fixB->GetFriction();

					// Calculate coefficient of friction for this contact
					// (average of both coefficients of friction)
					float c = contact->GetFriction();

					// Calculate force due to rolling resistance
					// F_rr = v * -c_rr
					b2Vec2 F_rr = -c * body->GetLinearVelocity();

					// Apply rolling resistance
					body->ApplyForceToCenter(F_rr, true);
				}
			}

			// Simple aerodynamic drag, calculated using the formula
			// Fad = -Cad * v|v|
			// Where Cad = the cross sectional area of the car
			// And v is the velocity
			// The cross sectional area is scaled down from world coordinates
			// by a factor of 1000 to produce nice results
			// A more complicated form of aerodynamic drag can be calculated
			// the mass density p of the air (dependent on its temperature and pressure),
			// and the drag coefficient Cd calculated using the object's geometry
			// The formula for this would be Fd = 0.5 * p * v|v| * Cd * Cad
			// But this is beyond the scope of this simulation
			b2Fixture* fixtures = body->GetFixtureList();
			if (fixtures != nullptr)
			{
				b2AABB aabb = fixtures->GetAABB(0);

				for (b2Fixture* fix = fixtures; fix; fix = fix->GetNext())
				{
					// The cross sectional area is estimated using the AABB of each shape
					aabb.Combine(fix->GetAABB(0));
				}

				b2Vec2 size = 0.001f * (aabb.upperBound - aabb.lowerBound);

				// Reverse size to get an approximation of cross sectional area
				float temp = size.x;
				size.x = size.y;
				size.y = temp;

				// Calculate v|v|
				b2Vec2 vsquared = body->GetLinearVelocity().Length() * body->GetLinearVelocity();

				// Calculate force of aerodynamic drag
				b2Vec2 Fad(-size.x * vsquared.x, -size.y * vsquared.y);

				// Apply force to body
				body->ApplyForceToCenter(Fad, true);
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

		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, VEHICLESIM_TITLE_FORMAT, getFPS());
		glfwSetWindowTitle(_window, title);
	}
}