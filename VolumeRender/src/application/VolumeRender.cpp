#include "application/VolumeRender.h"

#include <rendering/rendering_attributes.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

namespace vlr
{
	VolumeRender::VolumeRender(int32_t argc, char** argv)
		: Application(1920, 1080, "", true), _rot(0), _mesh("miku.md2", true),
		  _nearDepth(0.01f), _farDepth(10.0f), _scene(-1), _depth(0),
		  _saveTrees(true)
	{
		const double M_PI = 3.1415926535897;

		const int SCENE_COUNT = 8;
		const int OCTREE_MAX_DEPTH = 23;

		// Get tree filename
		_treeFilename = "out.tree";

		// Load scene # from argv
		if (argc > 1)
			_scene = atoi(argv[1]);

		if (argc > 2)
			_treeFilename = argv[2];

		if (argc > 3)
			_saveTrees = strcmp(argv[3], "dontsavetrees");

		if (_scene < 0 || _scene >= SCENE_COUNT)
			_scene = -1;

		while (_scene < 0 || _scene >= SCENE_COUNT)
		{
			// Do scene menu in console
			printf("Scenes:\n");
			printf("[0] tree file from argument 2\n");
			printf("[1] sphere\n");
			printf("[2] miku\n");
			printf("[3] miku + transparent sphere (outer r = 1.5 inner r = 1)\n");
			printf("[4] miku + transparent sphere (outer r = 1.5 inner hollow)\n");
			printf("[5] reflective + transparent teapot on checker board\n");
			printf("[6] reflective and transparent boxes on checker board\n");
			printf("[7] inverted reflected cube with teapot inside\n");

			printf("Select a scene: ");

			scanf("%d", &_scene);

			if (_scene < 0 || _scene >= SCENE_COUNT)
			{
				printf("Invalid scene\n");

				_scene = -1;
			}
		}

		// Load tree level
		if (_scene == 0 && argc > 3)
			_depth = atoi(argv[3]);
		else if (argc > 2)
			_depth = atoi(argv[2]);

		// Ask user for level if invalid
		while (_depth <= 0 || _depth > OCTREE_MAX_DEPTH)
		{
			printf("Select octree depth (1..%d (~8 is pretty fast to generate)): ", OCTREE_MAX_DEPTH);

			scanf("%d", &_depth);

			if (_depth <= 0 || _depth > OCTREE_MAX_DEPTH)
			{
				printf("Invalid depth\n");

				_depth = 0;
			}
		}

		// Set callbacks
		glfwSetFramebufferSizeCallback(_window, resize_callback);
 		glfwSetCursorPosCallback(_window, mouse_move_callback);
		glfwSetMouseButtonCallback(_window, mouse_callback);
		glfwSetKeyCallback(_window, key_callback);

		// Lock cursor
		_cursorLocked = true;
		glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		// Get cursor pos
		glfwGetCursorPos(_window, &_mouseX, &_mouseY);

		// Initialise framebuffer
		_fb.resize(getWidth(), getHeight());

		// Initialise camera
		int32_t width = getWidth();
		int32_t height = getHeight();
		float aspect = (float)height / (float)width;

		// Set camera rotation
		_camRot = glm::vec3(0.193f, 0.979f, 0.0f);

		_camera.setViewport(0, 0, width, height);
		_camera.perspective((float)(3.14159265358 / 2.0), aspect, 0.01f, 100.0f);
		_camera.translate(glm::vec3(1.13578f, 1.718106f, 2.026698f));

		// Rotate camera to initial rotation
		_camera.setRot(glm::quat());
		_camera.rotate(glm::vec3(_camRot.x, 0, 0));
		_camera.rotate(glm::vec3(0, _camRot.y, 0));

		// Clear rendering attributes
		memset(&rendering_attributes, 0, sizeof(rendering::rendering_attributes_t));
		
		// Generate/load data
		generate();

		// Set up rendering attributes for scene
		// Settings
		rendering_attributes.settings.enable_depth_copy = false;
		rendering_attributes.settings.enable_reflection = false;
		rendering_attributes.settings.enable_refraction = true;
		rendering_attributes.settings.enable_shadows = false;
		
		rendering_attributes.settings.refraction_mode = vlr::rendering::RefractionModes::DISCRETE;
		rendering_attributes.settings.refraction_discrete_step = 0.05f;

		// Clear colour
		rendering_attributes.clear_colour = glm::vec4(0.39f, 0.58f, 0.93f, 0.0f);

		// Light position and direction
		static glm::vec3 light_pos(2.02f, 2.06f, 2.28f);
		static glm::vec3 light_dir(0.314f, -0.217f, 0.074f);

		// Lights
		rendering_attributes.light_count = 1;

		rendering_attributes.ambient_colour = glm::vec3(rendering_attributes.clear_colour);

		vlr::rendering::light_t* dir_light = &rendering_attributes.lights[0];
		
		dir_light->type = rendering::LightTypes::POINT;
		dir_light->diffuse = glm::vec3(0.6f, 0.6f, 0.61f);
		dir_light->specular = glm::vec3(0.5f, 0.5f, 0.5f);
		dir_light->direction = light_dir;
		dir_light->position = light_pos;
		
		dir_light->constant_att = 1.0f;
		dir_light->linear_att = 0.0f;
		dir_light->quadratic_att = 0.0f;

		dir_light->exponent = 2.0f;
		dir_light->cutoff = M_PI / 4.0f;

		for (int32_t i = 1; i < 10; ++i)
		{
			dir_light = &rendering_attributes.lights[i];
		
			dir_light->type = rendering::LightTypes::POINT;
			dir_light->diffuse = glm::vec3(0, 0, 0.3f);
			dir_light->specular = glm::vec3(0.5f, 0.5f, 0.5f) * 0.0f;
			dir_light->direction = light_dir;
		
			dir_light->constant_att = 1.0f;
			dir_light->linear_att = 0.0f;
			dir_light->quadratic_att = 0.0f;
		}

		// Show window
		glfwShowWindow(_window);
	}
}
