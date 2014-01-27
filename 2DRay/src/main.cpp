#include "Ray2D.h"

using namespace vlr;

int main(int argc, char** argv)
{
	Ray2D application;

	// Loop while application running
	while (application.isRunning())
	{
		// Update and render application
		application.run();
	}

	return 0;
}
