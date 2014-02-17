#ifndef VEHICLESIM_ROCKETSYSTEM
#define VEHICLESIM_ROCKETSYSTEM

#include <Rocket/Core.h>

namespace vlr
{
	class RocketSystem
		: public Rocket::Core::SystemInterface
	{
	public:
		virtual float GetElapsedTime();
	};
}

#endif /* VEHICLESIM_ROCKETSYSTEM */
