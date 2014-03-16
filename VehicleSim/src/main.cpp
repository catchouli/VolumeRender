#include "VehicleSim.h"

#include "misc/GetterSetter.h"

#include <functional>

using namespace vlr;

class Test
{
public:
	Test() : t(999.0f) {}

	float getTest() const
	{
		return t;
	}
	 
	void setTest(const float& v)
	{
		t = v;
	}

	float t;
};

int main(int argc, char** argv)
{
	//Test test;
	//auto accessor = &Test::getTest;
	//auto mutator = &Test::setTest;
	//
	//GetterSetter<Test, float> funcGetter(&test, &Test::getTest, &Test::setTest);
	//GetterSetter<Test, float> pointGetter(&test, &test.t);

	//float funcVal = funcGetter.getValue();
	//float pointVal = pointGetter.getValue();

	//funcGetter.setValue(6);
	//funcVal = funcGetter.getValue();

	//pointGetter.setValue(5);
	//pointVal = pointGetter.getValue();

	//system("pause");

	VehicleSim app;

	while (app.isRunning())
	{
		app.run();
	}
}