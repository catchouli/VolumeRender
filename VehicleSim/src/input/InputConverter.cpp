#include "input/InputConverter.h"

namespace vlr
{
	InputConverter::InputConverter()
	{
		initKeycodes();
	}

	int InputConverter::convertMod(int glfwMods) const
	{
		int rocketMods = 0;

		// Get modifier states
		bool shift = false;
		bool ctrl = false;
		bool alt = false;
		bool super = false;
		
		shift = (glfwMods | GLFW_MOD_SHIFT) > 0;
		ctrl = (glfwMods | GLFW_MOD_CONTROL) > 0;
		alt = (glfwMods | GLFW_MOD_SHIFT) > 0;
		super = (glfwMods | GLFW_MOD_SUPER) > 0;

		// Convert to rocket
		if (shift)
			rocketMods |= Rocket::Core::Input::KM_SHIFT;
		if (ctrl)
			rocketMods |= Rocket::Core::Input::KM_CTRL;
		if (alt)
			rocketMods |= Rocket::Core::Input::KM_ALT;
		if (super)
			rocketMods |= Rocket::Core::Input::KM_META;
		
		return rocketMods;
	}

	Rocket::Core::Input::KeyIdentifier InputConverter::convertKeycode(int glfwKeycode) const
	{
		return keyMap[glfwKeycode];
	}
}