#include "tools/Tool.h"

#include "input/InputConverter.h"

namespace vlr
{
	void Tool::click_base(int button, int action, int mods)
	{
		if (_camera->getViewport().pointInViewport(_x, _y))
		{
			if (action == GLFW_PRESS)
			{
				_dragging = true;
				_startX = _x;
				_startY = _y;
			}
			else if (action == GLFW_RELEASE)
			{
				_dragging = false;
			}

			_mousedown = (action == GLFW_PRESS);

			// Call virtual function
			click(button, action, mods);
		}
	}

	void Tool::click(int button, int action, int mods)
	{

	}

	void Tool::mousemove_base(double x, double y, double dx, double dy)
	{
		_oldX = _x;
		_oldY = _y;
		_x = x;
		_y = y;

		mousemove(x, y, dx, dy);
	}

	void Tool::mousemove(double x, double y, double dx, double dy)
	{

	}

	void doInputSetButtons(std::vector<Gwen::Controls::Button*>& buttons, int key)
	{
		bool toggledOn = false;

		for (unsigned int i = 0; i < buttons.size(); ++i)
		{
			if (buttons[i]->GetToggleState())
			{
				toggledOn = true;
				break;
			}
		}

		if (toggledOn)
		{
			for (unsigned int i = 0; i < buttons.size(); ++i)
			{
				int* updateKey = buttons[i]->UserData.Get<int*>("updateKey");
				*updateKey = key;
				buttons[i]->SetText(InputConverter::translateCharToString(key));
				buttons[i]->SetToggleState(false);
			}
		}
	}

	void Tool::key_base(int key, int scancode, int action, int mods)
	{
		if (_enabled)
		{
			doInputSetButtons(_forwardButtons, key);
			doInputSetButtons(_reverseButtons, key);
		}

		this->key(key, scancode, action, mods);
	}

	void Tool::key(int key, int scancode, int action, int mods)
	{

	}

	void Tool::scroll_base(double x, double y)
	{
		scroll(x , y);
	}

	void Tool::scroll(double x, double y)
	{

	}

	void Tool::showOptions()
	{
		_tabButton->Show();
		_tabButton->GetPage()->Show();
		_tabButton->DoAction();
	}

	void Tool::hideOptions()
	{
		_tabButton->Hide();
		_tabButton->GetPage()->Hide();
		_tabButton->DoAction();
	}
}
