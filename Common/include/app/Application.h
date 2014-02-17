#ifndef VLR_COMMON_APPLICATION
#define VLR_COMMON_APPLICATION

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace vlr
{
	namespace common
	{
		class Application
		{
		public:
			Application(int width, int height, const char* title = "");
			~Application();

			void run();

			inline int getWidth() const;
			inline int getHeight() const;

			inline int getFPS() const;
			inline bool isRunning() const;

		protected:
			void setTitle(const char* title);

			void end();

			virtual void update(double dt) = 0;
			virtual void render() = 0;

			// Default glfw callbacks
			static void _default_key_callback(GLFWwindow* window, int key,
				int scancode, int action, int mods);

			GLFWwindow* _window;

		private:
			Application(const Application&);
			const Application& operator=(const Application&);

			bool _running;

			double _lastUpdate;
			double _lastFPSUpdate;
			int _frames;
			int _fps;
		};

		int Application::getWidth() const
		{
			int w, h;

			glfwGetWindowSize(_window, &w, &h);

			return w;
		}

		int Application::getHeight() const
		{
			int w, h;

			glfwGetWindowSize(_window, &w, &h);

			return h;
		}

		int Application::getFPS() const
		{
			return _fps;
		}

		bool Application::isRunning() const
		{
			return _running && !glfwWindowShouldClose(_window);
		}
	}
}

#endif /* VLR_COMMON_APPLICATION */