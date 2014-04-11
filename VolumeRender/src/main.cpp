#include "application/Ray2D.h"

#include "rendering/child_desc.h"

int main(int argc, char** argv)
{
	vlr::Ray2D app;

	while (app.isRunning())
	{
		app.run();
	}
}