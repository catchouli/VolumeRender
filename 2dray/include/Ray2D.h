#ifndef RAY2D_APPLICATION
#define RAY2D_APPLICATION

#include <app/SDLApplication.h>
#include <SDL2/SDL.h>
#include <stdio.h>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

namespace vlr
{
	class Ray2D
		: public common::SDLApplication
	{
	public:
		Ray2D();
		
		// Initialise fresh opengl context
		void initGL();

		// Overrides
		void initialise();
		void update(float dt);
		void render();
		void handleEvent(SDL_Event event);
	};
}

#endif /* 2DRAY_APPLICATION */
