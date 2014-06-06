#include "application/VolumeRender.h"

#include <rendering/child_desc.h>

int main(int argc, char** argv)
{
	float reflectivity = 1.0f;
	uint16_t reflectivity_packed;
	float reflectivity_unpacked;

	//vlr::rendering::pack_float(reflectivity, 16,

	//system("pause");
	//return 0;

	vlr::VolumeRender app(argc, argv);

	while (app.isRunning())
	{
		app.run();
	}
}