#ifndef VEHICLESIM_INPUTCONVERTER
#define VEHICLESIM_INPUTCONVERTER

#include <GLFW/glfw3.h>
#include <Rocket/Core/Input.h>

namespace vlr
{
	// Converts glfw keycodes to rocket keycodes
	class InputConverter
	{
	public:
		InputConverter();

		int convertMod(int glfwMod) const;
		Rocket::Core::Input::KeyIdentifier convertKeycode(int glfwKeycode) const;

	protected:
		InputConverter(const InputConverter&);

		void initKeycodes();

	private:
		static Rocket::Core::Input::KeyIdentifier keyMap[GLFW_KEY_LAST];
	};
};

#endif /* VEHICLESIM_INPUTCONVERTER */
