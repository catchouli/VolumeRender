#ifndef COMMON_SDLAPPLICATION
#define COMMON_SDLAPPLICATION

#include <SDL2/SDL.h>

namespace vlr
{
	namespace common
	{
		class SDLApplication
		{
		public:
			SDLApplication(int width, int height, const char* title = "");

			void run();

			static SDL_Window* createWindow(int width, int height, const char* title);
			static SDL_Renderer* createRenderer(SDL_Window* window);

			inline Uint32 getWidth() const;
			inline Uint32 getHeight() const;

			inline Uint32 getFPS() const;
			inline bool isRunning() const;

		protected:
			void setTitle(const char* title);

			void end();

			virtual void initialise() = 0;
			virtual void update(float dt) = 0;
			virtual void render() = 0;
			virtual void handleEvent(SDL_Event event) {};

		private:
			bool _running;

			Uint32 _lastUpdate;
			Uint32 _lastFPSUpdate;
			Uint32 _frames;
			Uint32 _fps;

			SDL_Window* _window;
			SDL_Renderer* _renderer;
		};

		Uint32 SDLApplication::getWidth() const
		{
			int w, h;

			SDL_GetWindowSize(_window, &w, &h);

			return w;
		}

		Uint32 SDLApplication::getHeight() const
		{
			int w, h;

			SDL_GetWindowSize(_window, &w, &h);

			return h;
		}

		Uint32 SDLApplication::getFPS() const
		{
			return _fps;
		}

		bool SDLApplication::isRunning() const
		{
			return _running;
		}
	}
}

#endif /* COMMON_SDLAPPLICATION */