#ifndef VLR_RENDERING_ATTRIBUTES
#define VLR_RENDERING_ATTRIBUTES

#include "rendering/Camera.h"

#include <glm/glm.hpp>
#include <cuda_runtime_api.h>

namespace vlr
{
	namespace rendering
	{
		// How many lights to use
		const int32_t MAX_LIGHTS = 10;

		namespace LightTypes
		{
			enum LightType
			{
				DIRECTIONAL,
				POINT,
				SPOT
			};
		}

		typedef LightTypes::LightType LightType;

		struct light_t
		{
			// Valid for all lights
			LightType type;

			// Colours
			glm::vec3 diffuse;
			glm::vec3 specular;

			// Valid for directional lights and spotlights
			glm::vec3 direction;

			// Valid for point lights and spotlights
			glm::vec3 position;

			// Attenuation for non directional lights
			float constant_att;
			float linear_att;
			float quadratic_att;

			// Valid for spotlights
			float cutoff;
			float exponent;
		};

		struct rendering_attributes_t
		{
			// Position, mvp matrix and viewport
			glm::vec3 origin;
			glm::mat4 mvp;
			viewport viewport;

			// The statically sized array of lights
			int32_t light_count;
			light_t lights[MAX_LIGHTS];

			// The ambient light in the scene
			glm::vec3 ambient_colour;
		};
	}
}

#endif /* VLR_RENDERING_ATTRIBUTES */