#include "app/SDLApplication.h"

#include <stdio.h>

namespace vlr
{
	namespace common
	{
		// Constructor
		// Create and initialise window and renderer
		SDLApplication::SDLApplication(int width, int height, const char* title)
			: _running(true), _lastUpdate(SDL_GetTicks()), _lastFPSUpdate(SDL_GetTicks()),
			  _fps(0), _frames(0)
		{
			// Create window
			_window = createWindow(width, height, title);

			// Create renderer
			_renderer = createRenderer(_window);
			
			// Allow application to initialise itself
			initialise();
		}

		// Handle message loop and call user defined update & render
		void SDLApplication::run()
		{
			SDL_Event event;

			// Handle events
			while (SDL_PollEvent(&event))
			{
				// Exit if user closes the window or presses esc
				if ((event.type == SDL_QUIT) ||
					(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
				{
					end();
				}
				else
				{
					// Pass event onto application
					handleEvent(event);
				}
			}

			// Calculate delta time
			int time = SDL_GetTicks();
			int dt = time - _lastUpdate;
			int fpsDt = time - _lastFPSUpdate;
			float dtSeconds = dt / 1000.0f;

			// Update timers
			_lastUpdate = time;

			// Update frame counter
			_frames++;

			// Update FPS if time > 1 second
			if (fpsDt >= 1000)
			{
				// Update FPS
				_fps = _frames;
				_frames = 0;
				_lastFPSUpdate = time;
			}

			// Call application update
			update(dtSeconds);

			// Call application render
			render();
			
			// Flip buffers
			SDL_RenderPresent(_renderer);
		}

		// Create a window with a specified title, width, and height
		SDL_Window* SDLApplication::createWindow(int width, int height, const char* title)
		{
			SDL_Window* window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
													width, height, SDL_WINDOW_OPENGL);

			if (window == nullptr)
			{
				fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
			}

			return window;
		}

		// Create a renderer for a window
		SDL_Renderer* SDLApplication::createRenderer(SDL_Window* window)
		{
			SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

			if (renderer == nullptr)
			{
				fprintf(stderr, "Could not create renderer: %s\n", SDL_GetError());
			}

			return renderer;
		}

		/* Protected */

		// Set window title
		void SDLApplication::setTitle(const char* title)
		{
			if (_window != nullptr)
				SDL_SetWindowTitle(_window, title);
		}

		// Exit application
		void SDLApplication::end()
		{
			_running = false;
		}
	}
}