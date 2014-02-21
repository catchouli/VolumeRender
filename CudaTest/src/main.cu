#include "CudaTest.h"

using namespace vlr;

int main(int argc, char** argv)
{
	CudaTest app;

	while (app.isRunning())
	{
		app.run();
	}
}