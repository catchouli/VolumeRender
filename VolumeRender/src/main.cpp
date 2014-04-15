#include "application/VolumeRender.h"

#include "rendering/child_desc.h"
#include "resources/Mesh.h"

int main(int argc, char** argv)
{
	vlr::rendering::Mesh mesh("miku.md2", true);

	{
		vlr::rendering::Mesh other(mesh);
	}

	vlr::VolumeRender app(argc, argv);

	while (app.isRunning())
	{
		app.run();
	}
}