#include "tools/Tool.h"

using namespace Gwen;
using namespace Gwen::Controls;

namespace vlr
{
	glm::vec2 Tool::worldSpace(float x, float y)
	{
		return glm::vec2(_camera->screenSpaceToWorld(x, y, 0));
	}

	glm::vec2 Tool::screenSpace(float x, float y)
	{
		return glm::vec2(_camera->worldSpaceToScreen(x, y, 0));
	}

	// Controls
	Gwen::Controls::Label* Tool::createLabel(const char* string, Gwen::Controls::Base* base, float x, float y)
	{
		// Create label
		Gwen::Controls::Label* label = new Gwen::Controls::Label(base);
		label->SetPos(x, y);
		label->SetText(string);
		label->SetWidth(250);

		return label;
	}
}