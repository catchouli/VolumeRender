#include "input/InputConverter.h"

#include <string.h>

using namespace Rocket::Core::Input;

namespace vlr
{
	KeyIdentifier InputConverter::keyMap[GLFW_KEY_LAST];

	void InputConverter::initKeycodes()
	{
		// Clear array
		memset(keyMap, 0,
			sizeof(KeyIdentifier) * GLFW_KEY_LAST);

		// Set values
		keyMap[GLFW_KEY_SPACE] = KI_SPACE;
		keyMap[GLFW_KEY_APOSTROPHE] = KI_OEM_7;
		keyMap[GLFW_KEY_COMMA] = KI_OEM_COMMA;
		keyMap[GLFW_KEY_MINUS] = KI_OEM_MINUS;
		keyMap[GLFW_KEY_PERIOD] = KI_OEM_PERIOD;
		keyMap[GLFW_KEY_SLASH] = KI_DIVIDE;

		keyMap[GLFW_KEY_0] = KI_0;
		keyMap[GLFW_KEY_1] = KI_1;
		keyMap[GLFW_KEY_2] = KI_2;
		keyMap[GLFW_KEY_3] = KI_3;
		keyMap[GLFW_KEY_4] = KI_4;
		keyMap[GLFW_KEY_5] = KI_5;
		keyMap[GLFW_KEY_6] = KI_6;
		keyMap[GLFW_KEY_7] = KI_7;
		keyMap[GLFW_KEY_8] = KI_8;
		keyMap[GLFW_KEY_9] = KI_9;

		keyMap[GLFW_KEY_SEMICOLON] = KI_OEM_1;
		keyMap[GLFW_KEY_EQUAL] = KI_OEM_NEC_EQUAL;

		keyMap[GLFW_KEY_A] = KI_A;
		keyMap[GLFW_KEY_B] = KI_B;
		keyMap[GLFW_KEY_C] = KI_C;
		keyMap[GLFW_KEY_D] = KI_D;
		keyMap[GLFW_KEY_E] = KI_E;
		keyMap[GLFW_KEY_F] = KI_F;
		keyMap[GLFW_KEY_G] = KI_G;
		keyMap[GLFW_KEY_H] = KI_H;
		keyMap[GLFW_KEY_I] = KI_I;
		keyMap[GLFW_KEY_J] = KI_J;
		keyMap[GLFW_KEY_K] = KI_K;
		keyMap[GLFW_KEY_L] = KI_L;
		keyMap[GLFW_KEY_M] = KI_M;
		keyMap[GLFW_KEY_N] = KI_N;
		keyMap[GLFW_KEY_O] = KI_O;
		keyMap[GLFW_KEY_P] = KI_P;
		keyMap[GLFW_KEY_Q] = KI_Q;
		keyMap[GLFW_KEY_R] = KI_R;
		keyMap[GLFW_KEY_S] = KI_S;
		keyMap[GLFW_KEY_T] = KI_T;
		keyMap[GLFW_KEY_U] = KI_U;	
		keyMap[GLFW_KEY_V] = KI_V;
		keyMap[GLFW_KEY_W] = KI_W;
		keyMap[GLFW_KEY_X] = KI_X;
		keyMap[GLFW_KEY_Y] = KI_Y;
		keyMap[GLFW_KEY_Z] = KI_Z;

		keyMap[GLFW_KEY_LEFT_BRACKET] = KI_OEM_4;
		keyMap[GLFW_KEY_BACKSLASH] = KI_OEM_5;
		keyMap[GLFW_KEY_RIGHT_BRACKET] = KI_OEM_6;
		keyMap[GLFW_KEY_GRAVE_ACCENT] = KI_OEM_3;

		keyMap[GLFW_KEY_ESCAPE] = KI_ESCAPE;
		keyMap[GLFW_KEY_ENTER] = KI_RETURN;
		keyMap[GLFW_KEY_TAB] = KI_TAB;
		keyMap[GLFW_KEY_BACKSPACE] = KI_BACK;
		keyMap[GLFW_KEY_INSERT] = KI_INSERT;
		keyMap[GLFW_KEY_DELETE] = KI_DELETE;

		keyMap[GLFW_KEY_RIGHT] = KI_RIGHT;
		keyMap[GLFW_KEY_LEFT] = KI_LEFT;
		keyMap[GLFW_KEY_DOWN] = KI_DOWN;
		keyMap[GLFW_KEY_UP] = KI_UP;

		keyMap[GLFW_KEY_PAGE_UP] = KI_PRIOR;
		keyMap[GLFW_KEY_PAGE_DOWN] = KI_NEXT;
		keyMap[GLFW_KEY_HOME] = KI_HOME;
		keyMap[GLFW_KEY_END] = KI_END;

		keyMap[GLFW_KEY_CAPS_LOCK] = KI_CAPITAL;
		keyMap[GLFW_KEY_SCROLL_LOCK] = KI_SCROLL;
		keyMap[GLFW_KEY_NUM_LOCK] = KI_NUMLOCK;
		keyMap[GLFW_KEY_PRINT_SCREEN] = KI_SNAPSHOT;

		keyMap[GLFW_KEY_PAUSE] = KI_PAUSE;

		keyMap[GLFW_KEY_F1] = KI_F1;
		keyMap[GLFW_KEY_F2] = KI_F2;
		keyMap[GLFW_KEY_F3] = KI_F3;
		keyMap[GLFW_KEY_F4] = KI_F4;
		keyMap[GLFW_KEY_F5] = KI_F5;
		keyMap[GLFW_KEY_F6] = KI_F6;
		keyMap[GLFW_KEY_F7] = KI_F7;
		keyMap[GLFW_KEY_F8] = KI_F8;
		keyMap[GLFW_KEY_F9] = KI_F9;
		keyMap[GLFW_KEY_F10] = KI_F10;
		keyMap[GLFW_KEY_F11] = KI_F11;
		keyMap[GLFW_KEY_F12] = KI_F12;
		keyMap[GLFW_KEY_F13] = KI_F13;
		keyMap[GLFW_KEY_F14] = KI_F14;
		keyMap[GLFW_KEY_F15] = KI_F15;
		keyMap[GLFW_KEY_F16] = KI_F16;
		keyMap[GLFW_KEY_F17] = KI_F17;
		keyMap[GLFW_KEY_F18] = KI_F18;
		keyMap[GLFW_KEY_F19] = KI_F19;
		keyMap[GLFW_KEY_F20] = KI_F20;
		keyMap[GLFW_KEY_F21] = KI_F21;
		keyMap[GLFW_KEY_F22] = KI_F22;
		keyMap[GLFW_KEY_F23] = KI_F23;
		keyMap[GLFW_KEY_F24] = KI_F24;

		keyMap[GLFW_KEY_KP_0] = KI_NUMPAD0;
		keyMap[GLFW_KEY_KP_1] = KI_NUMPAD1;
		keyMap[GLFW_KEY_KP_2] = KI_NUMPAD2;
		keyMap[GLFW_KEY_KP_3] = KI_NUMPAD3;
		keyMap[GLFW_KEY_KP_4] = KI_NUMPAD4;
		keyMap[GLFW_KEY_KP_5] = KI_NUMPAD5;
		keyMap[GLFW_KEY_KP_6] = KI_NUMPAD6;
		keyMap[GLFW_KEY_KP_7] = KI_NUMPAD7;
		keyMap[GLFW_KEY_KP_8] = KI_NUMPAD8;
		keyMap[GLFW_KEY_KP_9] = KI_NUMPAD9;
		
		keyMap[GLFW_KEY_KP_DECIMAL] = KI_DECIMAL;
		keyMap[GLFW_KEY_KP_DIVIDE] = KI_DIVIDE;
		keyMap[GLFW_KEY_KP_MULTIPLY] = KI_MULTIPLY;
		keyMap[GLFW_KEY_KP_SUBTRACT] = KI_SUBTRACT;
		keyMap[GLFW_KEY_KP_ADD] = KI_ADD;
		keyMap[GLFW_KEY_KP_DECIMAL] = KI_SEPARATOR;
		
		keyMap[GLFW_KEY_LEFT_SHIFT] = KI_LSHIFT;
		keyMap[GLFW_KEY_LEFT_CONTROL] = KI_LCONTROL;
		keyMap[GLFW_KEY_LEFT_ALT] = KI_LMENU;
		keyMap[GLFW_KEY_LEFT_SUPER] = KI_LWIN;
		keyMap[GLFW_KEY_RIGHT_SHIFT] = KI_RSHIFT;
		keyMap[GLFW_KEY_RIGHT_CONTROL] = KI_RCONTROL;
		keyMap[GLFW_KEY_RIGHT_ALT] = KI_RMENU;
		keyMap[GLFW_KEY_RIGHT_SUPER] = KI_RWIN;
	}
}
