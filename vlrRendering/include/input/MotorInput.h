#ifndef VLR_RENDERING_MOTORINPUT
#define VLR_RENDERING_MOTORINPUT

#include <GLFW/glfw3.h>
#include <Box2D/Box2D.h>

namespace vlr
{
	class MotorInput
	{
	public:
		MotorInput();
		void update(GLFWwindow* window, b2Joint* joint);

		inline bool getEnabled() const;

		inline void setEnabled(bool value);
		
		inline float getMaxForce() const;
		inline float getSpeed() const;
		
		inline void setMaxForce(float value);
		inline void setSpeed(float value);
		
		inline int getForwardKey() const;
		inline int getReverseKey() const;
		
		inline void setForwardKey(int value);
		inline void setReverseKey(int value);

	private:
		friend class Serialiser;

		friend class Tool;
		friend class SelectionTool;

		bool _enabled;
		float _maxForce, _speed;
		int _forwardButton, _reverseButton;
	};

	bool MotorInput::getEnabled() const
	{
		return _enabled;
	}

	void MotorInput::setEnabled(bool value)
	{
		_enabled = value;
	}
		
	float MotorInput::getMaxForce() const
	{
		return _maxForce;
	}

	float MotorInput::getSpeed() const
	{
		return _speed;
	}
		
	void MotorInput::setMaxForce(float value)
	{
		_maxForce = value;
	}

	void MotorInput::setSpeed(float value)
	{
		_speed = value;
	}
		
	int MotorInput::getForwardKey() const
	{
		return _forwardButton;
	}

	int MotorInput::getReverseKey() const
	{
		return _reverseButton;
	}
		
	void MotorInput::setForwardKey(int value)
	{
		_forwardButton = value;
	}

	void MotorInput::setReverseKey(int value)
	{
		_reverseButton = value;
	}
}

#endif /* VLR_RENDERING_MOTORINPUT */
