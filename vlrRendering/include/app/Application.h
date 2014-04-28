#ifndef VLR_RENDERING_APPLICATION
#define VLR_RENDERING_APPLICATION

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdint.h>

namespace vlr
{
	namespace rendering
	{
		class Application
		{
		public:
			Application(int32_t width, int32_t height, const char* title = "");
			~Application();

			void run();

			inline int32_t getWidth() const;
			inline int32_t getHeight() const;

			inline int32_t getFPS() const;
			inline bool isRunning() const;

		protected:
			void setTitle(const char* title);

			void end();

			virtual void update(double dt) = 0;
			virtual void render() = 0;

			// Default glfw callbacks
			static void _default_key_callback(GLFWwindow* window, int32_t key,
				int32_t scancode, int32_t action, int32_t mods);

			GLFWwindow* _window;

		private:
			Application(const Application&);
			const Application& operator=(const Application&);

			bool _running;

			double _lastUpdate;
			double _lastFPSUpdate;
			int32_t _frames;
			int32_t _fps;
		};

		int32_t Application::getWidth() const
		{
			int32_t w, h;

			glfwGetWindowSize(_window, &w, &h);

			return w;
		}

		int32_t Application::getHeight() const
		{
			int32_t w, h;

			glfwGetWindowSize(_window, &w, &h);

			return h;
		}

		int32_t Application::getFPS() const
		{
			return _fps;
		}

		bool Application::isRunning() const
		{
			return _running && !glfwWindowShouldClose(_window);
		}
	}
}

#endif /* VLR_RENDERING_APPLICATION */