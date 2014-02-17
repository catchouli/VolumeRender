#include "VehicleSim.h"

#include <Rocket/Core.h>

namespace vlr
{
	VehicleSim::VehicleSim()
		: Application(800, 600)
	{
		// Set callbacks
 		glfwSetCursorPosCallback(_window, mouse_move_callback);
		glfwSetMouseButtonCallback(_window, mouse_callback);
		glfwSetKeyCallback(_window, key_callback);

		// Initialise camera
		int width = getWidth();
		int height = getHeight();
		float aspect = (float)width / (float)height;

		_camera.setViewport(0, 0, width, height);
		_camera.perspective((float)(3.14159265358 / 2.0), aspect, 0.01f, 100.0f);
		_camera.translate(glm::vec3(0, 0, 10.0f));

		// Initialise Rocket
		Rocket::Core::SetSystemInterface(&_rocketSystem);
		Rocket::Core::SetRenderInterface(&_rocketRenderer);

		if (!Rocket::Core::Initialise())
		{
			fprintf(stderr, "Failed to initialise librocket\n");
		}
		
		// Create Rocket context
		_rocketContext = Rocket::Core::CreateContext("default",
			Rocket::Core::Vector2i(width, height));

		// Load document
		_document = _rocketContext->LoadDocument("assets/demo.rml");
		if (_document != nullptr)
			_document->Show();
	}

	VehicleSim::~VehicleSim()
	{
		_rocketContext->RemoveReference();
		_rocketContext = nullptr;
	}

	void VehicleSim::update(double dt)
	{
		const float MOVE_SPEED = 2.0f;

		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, "FPS: %d\n", getFPS());
		glfwSetWindowTitle(_window, title);

		// Update rocket
		_rocketContext->Update();
	}

	void VehicleSim::render()
	{
		// Clear screen
		glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		// Update opengl matrices
		_camera.updateGL();

		// Set up view
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Render
		//glBegin(GL_QUADS);
		//glColor3f(1.0f, 1.0f, 1.0f);
		//glVertex3f(0, 0, 0);
		//glVertex3f(1, 0, 0);
		//glVertex3f(1, 1, 0);
		//glVertex3f(0, 1, 0);
		//glEnd();

		// Reset matrices
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, getWidth(), getHeight(), 0, 0, 1000.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Render Rocket
		_rocketContext->Render();
	}

	void VehicleSim::mouse_callback(GLFWwindow* window, int button,
		int action, int mods)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Rocket::Core::Context* context = app->_rocketContext;
		const InputConverter& converter = app->_inputConverter;

		// Update rocket
		if (action == GLFW_PRESS)
			context->ProcessMouseButtonDown(button, converter.convertMod(mods));
		if (action == GLFW_RELEASE)
			context->ProcessMouseButtonUp(button, converter.convertMod(mods));
	}

	void VehicleSim::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Rocket::Core::Context* context = app->_rocketContext;

		// Update rocket
		context->ProcessMouseMove(x, y, 0);
	}

	void VehicleSim::key_callback(GLFWwindow* window, int key,
		int scancode, int action, int mods)
	{
		// Do default action (exit on esc)
		_default_key_callback(window, key, scancode, action, mods);

		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Rocket::Core::Context* context = app->_rocketContext;
		const InputConverter& converter = app->_inputConverter;

		// Update rocket
		if (action == GLFW_PRESS)
			context->ProcessKeyDown(converter.convertKeycode(key), converter.convertMod(mods));
		if (action == GLFW_RELEASE)
			context->ProcessKeyUp(converter.convertKeycode(key), converter.convertMod(mods));
	}
	
	void VehicleSim::scroll_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Rocket::Core::Context* context = app->_rocketContext;

		// Update rocket
		context->ProcessMouseWheel((int)y, 0);
	}
	
	void VehicleSim::scroll_callback(GLFWwindow* window,
			unsigned int codepoint)
	{
		// Get class instance
		VehicleSim* app = (VehicleSim*)glfwGetWindowUserPointer(window);
		Rocket::Core::Context* context = app->_rocketContext;

		// Update rocket
		context->ProcessTextInput(codepoint);
	}
}
