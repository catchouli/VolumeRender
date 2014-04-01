#include "application/Ray2D.h"

#include <functional>

using namespace vlr;

void doStuff(std::function<void(void)> f)
{
	f();
}

void func(int x)
{
	auto f = [x]() { printf("%d\n", x); };

	doStuff(f);
}

int main(int argc, char** argv)
{
	func(5);

	Ray2D app;

	while (app.isRunning())
	{
		app.run();
	}
}