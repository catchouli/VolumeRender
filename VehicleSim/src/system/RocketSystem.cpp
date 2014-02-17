#include "system/RocketSystem.h"

#include <GLFW/glfw3.h>

namespace vlr
{
	float RocketSystem::GetElapsedTime()
	{
		return (float)glfwGetTime();
	}
}
