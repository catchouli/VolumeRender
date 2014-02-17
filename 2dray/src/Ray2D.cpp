#include "Ray2D.h"

#include "rendering/Mesh.h"
#include "rendering/Octree.h"

#include <glm/glm.hpp>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <malloc.h>

namespace vlr
{
	std::ostream& operator<<(std::ostream& cout, glm::vec3 obj)
	{
		cout << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
		return cout;
	}

	int __float_as_int(float in) {
		union fi { int i; float f; } conv;
		conv.f = in;
		return conv.i;
	}

	float __int_as_float(int in) {
		union fi { int i; float f; } conv;
		conv.i = in;
		return conv.f;
	}

	int scale(int scale, unsigned int differing_bits)
	{
		int step_mask = 0;

		glm::vec3 pos(1, 1, 1);
		int CAST_STACK_DEPTH = 5;
		float scale_exp2 = (float)pow((double)2, scale - CAST_STACK_DEPTH);;

		if ((step_mask & 1) != 0) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.x + scale_exp2);
        if ((step_mask & 2) != 0) differing_bits |= __float_as_int(pos.y) ^ __float_as_int(pos.y + scale_exp2);
        if ((step_mask & 4) != 0) differing_bits |= __float_as_int(pos.z) ^ __float_as_int(pos.z + scale_exp2);
        scale = (__float_as_int((float)differing_bits) >> 23) - 127; // position of the highest bit
        scale_exp2 = __int_as_float((scale - CAST_STACK_DEPTH + 127) << 23); // exp2f(scale - s_max)

		return scale;
	}

	Ray2D::Ray2D()
		: Application(800, 600), _rot(0)
	{
		int price = 2;
		int cost = 40;

		for (int i = 1; i <= 163; ++i)
		{
			cost += price;
			price++;
		}

		for (int s = 1; s <= 140; ++s)
		{
			float p = (double)((100.0f*((140.0f-(float)s)/(101.0f-(float)s))));
			float x = (float)s + 0.5f * pow(p, 2) + 1.5f * ((100.0f*((float)(140-s)/(float)(101-s))));

			std::cout << "s: " << s << ", x: " << x << std::endl;
		}

		// Set callbacks
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

		// Generate cube
		mesh.createCube();

		// Initialise camera
		int width = getWidth();
		int height = getHeight();
		float aspect = (float)width / (float)height;

		_camera.setViewport(0, 0, width, height);
		_camera.perspective((float)(3.14159265358 / 2.0), aspect, 0.01f, 100.0f);
		_camera.translate(glm::vec3(0, 0, 10.0f));

		// Generate grid
		genGrid();

		// Generate octree from grid
		genOctree(_tree);
	}

	void Ray2D::update(double dt)
	{
		const float MOVE_SPEED = 2.0f;

		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, "FPS: %d\n", getFPS());
		glfwSetWindowTitle(_window, title);

		// Rotate cube
		_rot += (float)dt * 100.0f;

		// Handle movement
		if (_cursorLocked)
		{
			if (glfwGetKey(_window, GLFW_KEY_W))
			{
				_camera.translate(MOVE_SPEED * (float)dt * _camera.getForward());
			}
			if (glfwGetKey(_window, GLFW_KEY_S))
			{
				_camera.translate(-MOVE_SPEED * (float)dt * _camera.getForward());
			}
			if (glfwGetKey(_window, GLFW_KEY_A))
			{
				_camera.translate(MOVE_SPEED * (float)dt * _camera.getLeft());
			}
			if (glfwGetKey(_window, GLFW_KEY_D))
			{
				_camera.translate(-MOVE_SPEED * (float)dt * _camera.getLeft());
			}
			if (glfwGetKey(_window, GLFW_KEY_Q))
			{
				_camera.translate(glm::vec3(0, -MOVE_SPEED * (float)dt, 0));
			}
			if (glfwGetKey(_window, GLFW_KEY_E))
			{
				_camera.translate(glm::vec3(0, MOVE_SPEED * (float)dt, 0));
			}
		}
	}

	void Ray2D::render()
	{
		int i = glGetError();

		// Clear screen
		glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Set up viewport
		glViewport(0, 0, getWidth(), getHeight());

		// Set up culling & depth testing
		glFrontFace(GL_CW);
		glEnable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		
		// Update opengl matrices
		_camera.updateGL();
		//glMatrixMode(GL_PROJECTION);
		//glLoadIdentity();
		//glFrustum(-1.0f, 1.0f, 1.0f, -1.0, 1.0f, 100.0f);

		// Set up view
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(unprojected.x, unprojected.y, unprojected.z);
		//glRotatef(_rot, 1, 1.0f, 0);

		// Set up lighting
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);

		GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);

		GLfloat diffConst[] = { 1.0f, 1.0f, 1.0f, 1.0 };
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffConst);

		// Render grid from grid
		if (glfwGetKey(_window, GLFW_KEY_R))
		{
			glMatrixMode(GL_MODELVIEW);
			for (int x = 0; x < RAY2D_GRID_WIDTH; ++x)
			{
				for (int y = 0; y < RAY2D_GRID_HEIGHT; ++y)
				{
					for (int z = 0; z < RAY2D_GRID_DEPTH; ++z)
					{
						if (_grid[x][y][z])
						{
							glLoadIdentity();
							glTranslatef((GLfloat)x,
								(GLfloat)y, (GLfloat)z);
							mesh.render();
						}
					}
				}
			}
		}
		else
		{
			// Render octree
			renderOctreeGL(_tree);
		}
		
		glDisable (GL_LIGHTING);
		glLoadIdentity();

		glm::vec3 nextraypos = lastraypos + 1000.0f * lastraydir;

		glBegin(GL_LINES);
			glColor3f(0.0f, 0.0f, 0.0f);
			glVertex3f(lastraypos.x, lastraypos.y, lastraypos.z);
			glVertex3f(nextraypos.x, nextraypos.y, nextraypos.z);
		glEnd();
		
		if (glfwGetKey(_window, GLFW_KEY_F))
		{
			int width = getWidth();
			int height = getHeight();
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					int* pixel = _fb.getPointer() + y * width + x;

					if (raycastScreenPointGrid(x, y))
					{
						struct col
						{
							char r, g, b, a;
						};

						col colour;
						colour.r = (char)normal.x;
						colour.g = (char)normal.y;
						colour.b = (char)normal.z;
						colour.a = 0;

						pixel[0] = *(int*)&colour;
					}
					else
						pixel[0] = 0;
				}
			}

			_fb.render();
		}
		
		if (!glfwGetKey(_window, GLFW_KEY_G))
		{
			int width = getWidth();
			int height = getHeight();
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					int* pixel = _fb.getPointer() + y * width + x;

					if (raycastScreenPointOctree(x, y))
					{
						//struct col
						//{
						//	char r, g, b, a;
						//};

						//col colour;
						//colour.r = (char)normal.x;
						//colour.g = (char)normal.y;
						//colour.b = (char)normal.z;
						//colour.a = 0;

						//pixel[0] = *(int*)&colour;

						pixel[0] = -1;
					}
					else
						pixel[0] = 0;
				}
			}

			_fb.render();
		}
	}

	void Ray2D::genGrid()
	{
		// Initialise grid with random data
		long long length = RAY2D_GRID_WIDTH * RAY2D_GRID_HEIGHT * RAY2D_GRID_DEPTH;
		
		for (long long i = 0; i < length; ++i)
		{
			((int*)_grid)[i] = rand() % 10 < 3;
		}
	}

	void Ray2D::genOctree(common::Octree& tree)
	{
		const int MAX_DEPTH = 3;

		tree.root = new common::OctNode();
		tree.min = glm::vec3();
		tree.max = glm::vec3(RAY2D_GRID_WIDTH, RAY2D_GRID_HEIGHT,
			RAY2D_GRID_DEPTH);
		tree.depth = MAX_DEPTH;

		genNode(&tree.root, tree.min, tree.max, 0, MAX_DEPTH);
	}

	void Ray2D::genNode(common::OctNode** node, glm::vec3 min, glm::vec3 max, int depth, int maxDepth)
	{
#define ARR_IDX(x, y, z, width, height) x * width * height + y * width + z

		if (depth > maxDepth)
			return;

		// Check if this node contains anything
		for (int x = (int)min.x; x < (int)max.x; ++x)
		{
			for (int y = (int)min.y; y < (int)max.y; ++y)
			{
				for (int z = (int)min.z; z < (int)max.z; ++z)
				{
					if (_grid[x][y][z])
					{
						(*node) = new common::OctNode();

						break;
					}
				}

				if (*node != nullptr)
					break;
			}

			if (*node != nullptr)
				break;
		}

		if (*node == nullptr)
			return;

		// Create child nodes
		glm::vec3 halfwidth = 0.5f * glm::vec3(max.x - min.x, 0, 0);
		glm::vec3 halfheight = 0.5f * glm::vec3(0, max.y - min.y, 0);
		glm::vec3 halfdepth = 0.5f * glm::vec3(0, 0, max.z - min.z);

		for (int x = 0; x < 2; ++x)
		{
			for (int y = 0; y < 2; ++y)
			{
				for (int z = 0; z < 2; ++z)
				{
					// Calculate new bounding box
					glm::vec3 newMin = min + (float)x * halfwidth +
						(float)y * halfheight + (float)z * halfdepth;
					glm::vec3 newMax = newMin + halfwidth + halfheight
						+ halfdepth;

					int idx = x*4 + y*2 + z;

					common::OctNode** nextChild = &((*node)->children[idx]);
					
					genNode(nextChild, newMin, newMax, depth+1, maxDepth);

					int i = 0;
				}
			}
		}
		
		(*node)->leaf = true;
		for (int i = 0; i < 8; ++i)
		{
			if ((*node)->children[i] != nullptr)
			{
				(*node)->leaf = false;
				return;
			}
		}
	}

	void Ray2D::renderOctreeGL(common::Octree tree)
	{
		renderNodeGL(tree.root, tree.min, tree.max);
	}

	void Ray2D::renderNodeGL(common::OctNode* node, glm::vec3 min,
		glm::vec3 max)
	{
		if (node->leaf)
		{
			glm::vec3 scale = max - min;
			
			glLoadIdentity();
			glTranslatef(min.x, min.y, min.z);
			glScalef(scale.x, scale.y, scale.z);
			mesh.render();
			return;
		}

		glm::vec3 halfwidth = 0.5f * glm::vec3(max.x - min.x, 0, 0);
		glm::vec3 halfheight = 0.5f * glm::vec3(0, max.y - min.y, 0);
		glm::vec3 halfdepth = 0.5f * glm::vec3(0, 0, max.z - min.z);

		for (int x = 0; x < 2; ++x)
		{
			for (int y = 0; y < 2; ++y)
			{
				for (int z = 0; z < 2; ++z)
				{
					// Calculate new bounding box
					glm::vec3 newMin = min + (float)x * halfwidth +
						(float)y * halfheight + (float)z * halfdepth;
					glm::vec3 newMax = newMin + halfwidth + halfheight
						+ halfdepth;

					int idx = x*4 + y*2 + z;

					// Get node
					common::OctNode* newNode = node->children[idx];
					if (newNode != nullptr)
						renderNodeGL(newNode, newMin, newMax);

				}
			}
		}
	}

	template <typename T>
	void swap(T& x, T& y)
	{
		T temp = x;
		x = y;
		y = temp;
	}

	Ray2D::Ray Ray2D::screenPointToRay(int x, int y)
	{
		Ray ray;

		glm::vec3 camPos = _camera.getPos();
		ray.origin = camPos;

		// Convert x and y to viewport space
		float width = getWidth();
		float height = getHeight();
		float normx = x / width;
		float normy = y / height;

		float xvs = normx * 2.0f - 1.0f;
		float yvs = normy * 2.0f - 1.0f;
		float zvs = 0.01f;

		glm::vec4 viewpoint(xvs, yvs, zvs, 1);
		glm::vec4 world = viewpoint * _camera.getMVP();
		glm::vec3 dir = glm::normalize(glm::vec3(world));

		ray.direction = dir;

		return ray;
	}

	bool Ray2D::raycastScreenPointGrid(int x, int y)
	{
		Ray2D::Ray ray = screenPointToRay(x, y);

		lastraypos = ray.origin;
		lastraydir = ray.direction;

		return raycastGrid(ray.origin, ray.direction);
	}

	bool Ray2D::raycastGrid(glm::vec3 origin, glm::vec3 direction)
	{
#define ray2d_min(x, y) (x < y ? x : y)
#define ray2d_min3(x, y, z) (ray2d_min(x, y) < z ? ray2d_min(x, y) : z)
#define ray2d_max(x, y) (x > y ? x : x)
#define ray2d_max3(x, y, z) (ray2d_max(x, y) > z ? ray2d_max(x, y) : z)

#define ray2d_t(x, i) ((x - origin.i) / direction.i)
		
		int xmin = 0, ymin = 0, zmin = 0;
		int xmax = RAY2D_GRID_WIDTH;
		int ymax = RAY2D_GRID_HEIGHT;
		int zmax = RAY2D_GRID_DEPTH;
		
		int dxsign = (direction.x > 0 ? 1 : -1);
		int dysign = (direction.y > 0 ? 1 : -1);
		int dzsign = (direction.z > 0 ? 1 : -1);
		
		float txmin = ray2d_t(xmin, x);
		float tymin = ray2d_t(ymin, y);
		float tzmin = ray2d_t(zmin, z);

		float txmax = ray2d_t(xmax, x);
		float tymax = ray2d_t(ymax, y);
		float tzmax = ray2d_t(zmax, z);

		if (txmin > txmax)
			swap(txmin, txmax);

		if (tymin > tymax)
			swap(tymin, tymax);

		if (tzmin > tzmax)
			swap(tzmin, tzmax);

		float tmin = ray2d_max3(txmin, tymin, tzmin);
		float tmax = ray2d_min3(txmax, tymax, tzmax);

		float t = 0;

		glm::vec3 pos = origin + t * direction;

		int x = (int)pos.x;
		int y = (int)pos.y;
		int z = (int)pos.z;

		while (t < tmax)
		{
			//int nextx = x+dxsign;
			//int nexty = y+dysign;
			//int nextz = z+dzsign;
			//
			//float tx = ray2d_t(nextx, x);
			//float ty = ray2d_t(nexty, y);
			//float tz = ray2d_t(nextz, z);

			//if (tx <= ty && tx <= tz)
			//{
			//	t = tx;
			//	x = nextx;
			//}
			//else if (ty <= tx && ty <= tz)
			//{
			//	t = ty;
			//	y = nexty;
			//}
			//else if (tz <= tx && tz <= ty)
			//{
			//	t = tz;
			//	z = nextz;
			//}
			//else
			//{
			//	std::cout << "This should never happen" << std::endl;
			//}

			//if (_grid[x][y][z] != 0)
			//	return true;

			/* Original method */
			t += 0.1f;
			pos = origin + t * direction;
			
			int oldX = x;
			int oldY = y;
			int oldZ = z;

			x = (int)pos.x;
			y = (int)pos.y;
			z = (int)pos.z;

			if (x >= 0 && y >= 0 && z >= 0 &&
				x < xmax && y < ymax && z < zmax && _grid[x][y][z])
			{
				lastx = x;
				lasty = y;
				lastz = z;

				//_grid[x][y][z] = 0;
				if (oldX < x)
					normal = glm::vec3(255, 0, 0);
				else if (oldX > x)
					normal = glm::vec3(255, 0, 0);
				else if (oldY < y)
					normal = glm::vec3(0, 255, 0);
				else if (oldY > y)
					normal = glm::vec3(0, 255, 0);
				else if (oldZ < z)
					normal = glm::vec3(0, 0, 255);
				else if (oldZ > z)
					normal = glm::vec3(0, 0, 255);
				else
					std::cout << "This should never happen";

				return true;
			}
		}

		return false;
	}

	bool Ray2D::raycastScreenPointOctree(int x, int y)
	{
		Ray point = screenPointToRay(x, y);

		return raycastOctree(point.origin, point.direction);
	}

	template <typename T>
	T sign(T value)
	{
		if (value < 0)
			return -1;
		else
			return 1;
	}

	template <typename T>
	T abs(T value)
	{
		if (value > 0)
			return value;
		else
			return -value;
	}

	bool Ray2D::raycastOctree(glm::vec3 origin, glm::vec3 direction)
	{
		struct Pos
		{
			int x, y, z;
		};

		struct StackEntry
		{
			common::OctNode parent;
			Pos pos;
		};

		StackEntry* stack = new StackEntry[_tree.depth];

		const float minf = 0.0001f;
		
		if (abs(direction.x) < minf) direction.x = sign(direction.x) * minf;
		if (abs(direction.y) < minf) direction.y = sign(direction.y) * minf;
		if (abs(direction.z) < minf) direction.z = sign(direction.z) * minf;

		int xmin = _tree.min.x;
		int ymin = _tree.min.y;
		int zmin = _tree.min.z;
		int xmax = _tree.max.x;
		int ymax = _tree.max.y;
		int zmax = _tree.max.z;
		
		int dxsign = (direction.x > 0 ? 1 : -1);
		int dysign = (direction.y > 0 ? 1 : -1);
		int dzsign = (direction.z > 0 ? 1 : -1);
		
		float txmin = ray2d_t(xmin, x);
		float tymin = ray2d_t(ymin, y);
		float tzmin = ray2d_t(zmin, z);

		float txmax = ray2d_t(xmax, x);
		float tymax = ray2d_t(ymax, y);
		float tzmax = ray2d_t(zmax, z);

		//// Make sure this hits the root
		//// Calculate the min and max positions of the root
		//glm::vec3 min(xmin, ymin, zmin);
		//glm::vec3 max(xmax, ymax, zmax);

		//// Calculate min and max points of ray at boundaries
		//glm::vec3 raymin(txmin, tymin, tzmin);
		//glm::vec3 raymax(txmax, tymax, tzmax);
		//
		//raymin *= direction;
		//raymax *= direction;

		//raymin += origin;
		//raymax += origin;
		
		//// Check they're within the boundaries of the root node
		//if (raymin.x < min.x || raymin.y < min.y || raymin.z < min.z ||
		//	raymax.x > max.x || raymax.y > max.y || raymax.z > max.z)
		//	return false;
		
		if (txmin > txmax)
			swap(txmin, txmax);

		if (tymin > tymax)
			swap(tymin, tymax);

		if (tzmin > tzmax)
			swap(tzmin, tzmax);

		float tmin = ray2d_max3(txmin, tymin, tzmin);
		float tmax = ray2d_min3(txmax, tymax, tzmax);

		common::OctNode* parent = _tree.root;

		auto getpos = [] (char idx) -> Pos
		{
			Pos pos;
			
			pos.x = (idx & 1) == 1;
			pos.y = (idx & 2) == 2;
			pos.z = (idx & 4) == 4;

			return pos;
		};

		auto getidx = [] (Pos pos) -> char
		{
			char idx = 0;

			idx |= pos.x;
			idx |= pos.y << 1;
			idx |= pos.z << 2;

			return idx;
		};

		auto advance = [&getidx] (glm::vec3& min, glm::vec3& max,
			glm::vec3& origin, glm::vec3& direction, int tmax, Pos& pos)
			-> char
		{
			char idx = 0;
			
			idx = getidx(pos);

			// Evaluate t at max
			float tcx = ray2d_t(max.x, x);
			float tcy = ray2d_t(max.y, y);
			float tcz = ray2d_t(max.z, z);

			// Compare against tmax to get new positions
			idx ^= (tcx == tmax) << 0;
			idx ^= (tcy == tmax) << 1;
			idx ^= (tcz == tmax) << 2;

			return idx;
		};

		auto push = [] (glm::vec3& min, glm::vec3& max,
			glm::vec3& origin, glm::vec3& direction, int tmin) -> char
		{
			char idx = 0;

			// Evaluate t at centre
			glm::vec3& centre = min + (max - min) * 0.5f;
			float tcx = ray2d_t(centre.x, x);
			float tcy = ray2d_t(centre.y, y);
			float tcz = ray2d_t(centre.z, z);

			// Compare against tmax to get new positions
			idx |= (tcx == tmin) << 0;
			idx |= (tcy == tmin) << 1;
			idx |= (tcz == tmin) << 2;

			return idx;
		};

		float t = tmin;

		for (int i = 0; i < 8; ++i)
		{
			Pos pos = getpos(i);
			char idx = getidx(pos);
			std::cout << "idx: " << (int)idx << ", pos: " << pos.x << ", " << pos.y << ", " <<
				pos.z << std::endl;
		}
				
		while (t < tmax)
		{


			int i = 0;
		}

		delete[] stack;
		return false;
	}

	void Ray2D::mouse_callback(GLFWwindow* window, int button,
		int action, int mods)
	{
		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		common::Camera& cam = ray2d->_camera;

		if (!ray2d->_cursorLocked && action == GLFW_PRESS)
		{
			bool hit = ray2d->raycastScreenPointGrid((int)(ray2d->_mouseX),
				(int)(ray2d->getHeight() - ray2d->_mouseY));

			if (hit)
			{
				std::cout << "Yay! hit" << std::endl;
				(ray2d->_grid)[ray2d->lastx][ray2d->lasty][ray2d->lastz] = 0;
			}
			else
			{
				std::cout << "Aww! miss" << std::endl;
			}
		}
	}

	void Ray2D::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		common::Camera& cam = ray2d->_camera;

		// Calculate difference
		double diffX = x - ray2d->_mouseX;
		double diffY = y - ray2d->_mouseY;
		
		// Update mouse pos
		ray2d->_mouseX = x;
		ray2d->_mouseY = y;

		// Camera rotation x and y
		static float camRotX = 0;
		static float camRotY = 0;

		if (ray2d->_cursorLocked)
		{
			// Update camera rotation
			camRotX += diffY * 0.001f;
			camRotY += diffX * 0.001f;

			// Rotate camera with mouse movement
			// Calculate rotation
			cam.setRot(glm::quat());
			cam.rotate(glm::vec3(camRotX, 0, 0));
			cam.rotate(glm::vec3(0, camRotY, 0));
		}
	}

	void Ray2D::key_callback(GLFWwindow* window, int key,
		int scancode, int action, int mods)
	{
		// Do default action (exit on esc)
		_default_key_callback(window, key, scancode, action, mods);

		// Get class instance
		Ray2D* ray2d = (Ray2D*)glfwGetWindowUserPointer(window);
		common::Camera& cam = ray2d->_camera;

		// Regenerate grid
		if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
			ray2d->genGrid();

		// Lock/unlock cursor & disable input
		if (key == GLFW_KEY_L && action == GLFW_PRESS)
		{
			ray2d->_cursorLocked = !ray2d->_cursorLocked;
			if (ray2d->_cursorLocked)
			{
				glfwSetInputMode(ray2d->_window,
					GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else
			{
				glfwSetInputMode(ray2d->_window,
					GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}
}
