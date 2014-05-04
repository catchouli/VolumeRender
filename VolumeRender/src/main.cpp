#include "application/VolumeRender.h"

#include "rendering/child_desc.h"
#include "resources/Mesh.h"

int32_t main(int32_t argc, char** argv)
{
	vlr::VolumeRender app(argc, argv);

	while (app.isRunning())
	{
		app.run();
	}
}