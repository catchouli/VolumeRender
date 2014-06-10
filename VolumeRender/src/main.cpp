#include "application/VolumeRender.h"

int main(int argc, char** argv)
{
	vlr::VolumeRender app(argc, argv);

	while (app.isRunning())
	{
		app.run();
	}
}