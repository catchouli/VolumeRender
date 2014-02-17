#include "VehicleSim.h"

using namespace vlr;

int main(int argc, char** argv)
{
	VehicleSim app;

	while (app.isRunning())
	{
		app.run();
	}
}