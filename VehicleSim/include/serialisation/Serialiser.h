#ifndef VEHICLESIM_SERIALISER
#define VEHICLESIM_SERIALISER

#include <string>

class b2World;

namespace vlr
{
	class Serialiser
	{
	public:
		static std::string serialiseWorld(const b2World* world);
		static void deserialiseWorld(b2World* world, std::string string);
		static void destroyWorld(b2World* world);

	protected:
		Serialiser();
		Serialiser(const Serialiser&);
	};
}

#endif /* VEHICLESIM_SERIALISER */
