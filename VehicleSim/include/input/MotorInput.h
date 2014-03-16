#ifndef VEHICLESIM_MOTORINPUT
#define VEHICLESIM_MOTORINPUT

#include <GLFW/glfw3.h>
#include <Box2D/Box2D.h>

namespace vlr
{
	struct MotorInput
	{
		MotorInput()
			: maxForce(0), speed(0),
			forwardButton(GLFW_KEY_UNKNOWN), reverseButton(GLFW_KEY_UNKNOWN)
		{

		}

		void update(GLFWwindow* window, b2Joint* joint)
		{
			bool forwardKey = glfwGetKey(window, forwardButton) != 0;
			bool reverseKey = glfwGetKey(window, reverseButton) != 0;

			bool enableMotor = forwardKey != reverseKey;

			switch (joint->GetType())
			{
			case b2JointType::e_revoluteJoint:
				{
					b2RevoluteJoint* specJoint = (b2RevoluteJoint*)joint;

					specJoint->EnableMotor(enableMotor);

					if (enableMotor)
					{
						specJoint->SetMaxMotorTorque(maxForce);
						specJoint->SetMotorSpeed(speed * (reverseKey ? -1.0f : 1.0f));
					}
				}
				break;
			case b2JointType::e_wheelJoint:
				{
					b2WheelJoint* specJoint = (b2WheelJoint*)joint;

					specJoint->EnableMotor(enableMotor);

					if (enableMotor)
					{
						specJoint->SetMaxMotorTorque(maxForce);
						specJoint->SetMotorSpeed(speed * (reverseKey ? -1.0f : 1.0f));
					}
				}
				break;
			case b2JointType::e_prismaticJoint:
				{
					b2PrismaticJoint* specJoint = (b2PrismaticJoint*)joint;

					specJoint->EnableMotor(enableMotor);

					if (enableMotor)
					{
						specJoint->SetMaxMotorForce(maxForce);
						specJoint->SetMotorSpeed(speed * (reverseKey ? -1.0f : 1.0f));
					}
				}
				break;
			default:
				return;
			}
		}

		float maxForce, speed;
		int forwardButton, reverseButton;
	};
}

#endif /* VEHICLESIM_MOTORINPUT */
