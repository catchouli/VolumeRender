#ifndef VEHICLESIM_SERIALISER
#define VEHICLESIM_SERIALISER

#include <string>

class b2World;

namespace vlr
{
	class VehicleSim;

	class Serialiser
	{
	public:
		static std::string serialiseWorld(const VehicleSim* vehicleSim,
			const b2World* world);
		static void deserialiseWorld(VehicleSim* vehicleSim, b2World* world,
			std::string string);
		static void destroyWorld(VehicleSim* vehicleSim, b2World* world);

	protected:
		Serialiser();
		Serialiser(const Serialiser&);
	};
}

#endif /* VEHICLESIM_SERIALISER */
