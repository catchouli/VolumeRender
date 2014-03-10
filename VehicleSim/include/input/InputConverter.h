#ifndef VEHICLESIM_INPUTCONVERTER
#define VEHICLESIM_INPUTCONVERTER

#include <GLFW/glfw3.h>
#include <Gwen/Controls/Canvas.h>

namespace vlr
{
	// Converts glfw keycodes to rocket keycodes
	class InputConverter
	{
	public:
		static unsigned char translateKeyCode(int keycode)
	{
		// TODO: implement if keyboard input necessary
		 switch (keycode)
		 {
		 case GLFW_KEY_BACKSPACE:
			 return Gwen::Key::Backspace;
		 case GLFW_KEY_ENTER:
			 return Gwen::Key::Return;
		 case GLFW_KEY_ESCAPE:
			 return Gwen::Key::Escape;
		 case GLFW_KEY_TAB:
			 return Gwen::Key::Tab;
		 case GLFW_KEY_SPACE:
			 return Gwen::Key::Space;
		 case GLFW_KEY_UP:
			 return Gwen::Key::Up;
		 case GLFW_KEY_DOWN:
			 return Gwen::Key::Down;
		 case GLFW_KEY_LEFT:
			 return Gwen::Key::Left;
		 case GLFW_KEY_RIGHT:
			 return Gwen::Key::Right;
		 case GLFW_KEY_HOME:
			 return Gwen::Key::Home;
		 case GLFW_KEY_END:
			 return Gwen::Key::End;
		 case GLFW_KEY_DELETE:
			 return Gwen::Key::Delete;
		 case GLFW_KEY_LEFT_CONTROL:
			 return Gwen::Key::Control;
		 case GLFW_KEY_LEFT_ALT:
			 return Gwen::Key::Alt;
		 case GLFW_KEY_LEFT_SHIFT:
			 return Gwen::Key::Shift;
		 case GLFW_KEY_RIGHT_CONTROL:
			 return Gwen::Key::Control;
		 case GLFW_KEY_RIGHT_ALT:
			 return Gwen::Key::Alt;
		 case GLFW_KEY_RIGHT_SHIFT:
			 return Gwen::Key::Shift;
		 default:
			 return Gwen::Key::Invalid;
		 }
	}

	protected:
		InputConverter();
		InputConverter(const InputConverter&);
	};
};

#endif /* VEHICLESIM_INPUTCONVERTER */
