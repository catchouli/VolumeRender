#include "CudaTest.h"

#include <iostream>

#include <cuda.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

namespace vlr
{
	template <typename T>
	__device__ void swap(T a, T b)
	{
		T temp = a;
		a = b;
		b = a;
	}

	template <typename T>
	__device__ inline void swapminmax(T& a, T& b)
	{
		T tempa, tempb;
		tempa = a;
		tempb = b;

		a = fminf(tempa, tempb);
		b = fmaxf(tempa, tempb);
	}

	__device__ float3 operator+(float3 f31, float3 f32)
	{
		f31.x += f32.x;
		f31.y += f32.y;
		f31.z += f32.z;

		return f31;
	}

	__device__ float3 operator*(float3 f3, float f)
	{
		f3.x *= f;
		f3.y *= f;
		f3.z *= f;

		return f3;
	}

	__global__ void raycast(int* framebuffer, int fbw, int fbh,
		int* grid, glm::mat4* mvp, glm::vec3* origin, int w,
		int h, int d)
	{
		int x = blockIdx.x;
		int y = blockIdx.y;

		const float PI = 3.1415926535f;
		const float FOV = PI / 4.0f;

		// Calculate ray origin & direction
		float3 o = {0, 0, 30.0f};
		float3 dir;
		
		dir.x = tanf(FOV) * ((2 * x - fbw) / fbw);
		dir.y = tanf(FOV) * ((2 * y - fbh) / fbh);
		dir.z = 1;

		float magnitude = dir.x * dir.x +
			dir.y * dir.y + dir.z * dir.z;
		
		dir.x /= magnitude;
		dir.y /= magnitude;
		dir.z /= magnitude;

		// Calculate coefficients for t value calculation
		int xmin = 0, ymin = 0, zmin = 0;
		int xmax = w;
		int ymax = h;
		int zmax = d;

		float xcoeff = 1.0f / dir.x;
		float ycoeff = 1.0f / dir.y;
		float zcoeff = 1.0f / dir.z;
		
		float xoffset = -(o.x / dir.x);
		float yoffset = -(o.y / dir.y);
		float zoffset = -(o.z / dir.z);
		
		float txmin = xmin * xcoeff + xoffset;
		float tymin = ymin * ycoeff + yoffset;
		float tzmin = zmin * zcoeff + zoffset;

		float txmax = xmax * xcoeff + xoffset;
		float tymax = ymax * ycoeff + yoffset;
		float tzmax = zmax * zcoeff + zoffset;

		swapminmax(txmin, txmax);
		swapminmax(tymin, tymax);
		swapminmax(tzmin, tzmax);
		
		float tmin = fmaxf(fmaxf(txmin, tymin), tzmin);
		float tmax = fminf(fminf(txmax, tymax), tzmax);

		float t = tmin;

		float3 pos = o + dir * t;

		int cx = (int)pos.x;
		int cy = (int)pos.y;
		int cz = (int)pos.z;
		
		int oldx = cx;
		int oldy = cy;
		int oldz = cz;

		framebuffer[y * fbw + x] = 0;

		while (t <= tmax)
		{
			t += 0.1f;
			pos = o + dir * t;

			oldx = cx;
			oldy = cy;
			oldz = cz;
				
			cx = (int)pos.x;
			cy = (int)pos.y;
			cz = (int)pos.z;

			if (cx >= 0 && cy >= 0 && cz >= 0 &&
				cx < xmax && cy < ymax && cz < zmax &&
				grid[cz * w * h + cy * w + cx] != 0)
			{
				framebuffer[y * fbw + x] = 0xFFFFFFFF;

				return;
			}
		}

		return;
	}

	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
	{
	   if (code != cudaSuccess) 
	   {
		  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			system("pause");
		  if (abort) exit(code);
	   }
	}

	CudaTest::CudaTest()
		: Application(800, 600)
	{
		// Set callbacks
 		glfwSetCursorPosCallback(_window, mouse_move_callback);
		glfwSetMouseButtonCallback(_window, mouse_callback);
		glfwSetKeyCallback(_window, key_callback);

		// Get width and height
		_width = getWidth();
		_height = getHeight();

		// Initialise camera
		float aspect = (float)_width / (float)_height;

		_camera.setViewport(0, 0, _width, _height);
		_camera.perspective((float)(3.14159265358 / 2.0), aspect, 0.01f, 100.0f);
		_camera.translate(glm::vec3(0, 0, 10.0f));

		_mvp = _camera.getMVP();

		// Generate grid
		genGrid();

		// Allocate memory on gpu
		gpuErrchk(cudaMalloc(&_mvpGpu, sizeof(glm::mat4)));
		gpuErrchk(cudaMalloc(&_originGpu, sizeof(glm::vec3)));
		gpuErrchk(cudaMalloc(&_gridGpu, sizeof(_grid)));

		// Copy grid to gpu
		gpuErrchk(cudaMemcpy(_gridGpu, _grid, sizeof(_grid),
			cudaMemcpyHostToDevice));

		// Create pbo
		glGenBuffers(1, &_pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, _width*_height*4*sizeof(GLubyte), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// Create opengl texture for cuda framebuffer
		glGenTextures(1, &_texid);
		glBindTexture(GL_TEXTURE_2D, _texid);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);

		gpuErrchk(cudaGraphicsGLRegisterBuffer(&_glFb, _pbo, cudaGraphicsRegisterFlagsNone));
	}

	void CudaTest::update(double dt)
	{
		// Set window title
		const int TITLE_LEN = 1024;
		char title[1024];
		sprintf(title, "FPS: %d\n", getFPS());
		glfwSetWindowTitle(_window, title);

		// Get mvp matrix
		_mvp = _camera.getMVP();

		// Get cam pos
		glm::vec3 camPos = _camera.getPos();

		// Copy MVP and cam pos to gpu
		gpuErrchk(cudaMemcpy(_mvpGpu, &_mvp, sizeof(glm::mat4),
			cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(_originGpu, &camPos, sizeof(glm::vec3),
			cudaMemcpyHostToDevice));
	}

	void CudaTest::render()
	{
		void* ptr;
		size_t size;

		int col = (int)(glfwGetTime() * 100.0);

		// Map PBO to cuda
		gpuErrchk(cudaGraphicsMapResources(1, &_glFb, 0));

		//// Get a device pointer to it
		gpuErrchk(cudaGraphicsResourceGetMappedPointer(&ptr,
			&size, _glFb));

		// Run kernel
		dim3 grid_size(_width, _height);
		raycast<<<grid_size, 1>>>((int*)ptr, _width, _height,
			_gridGpu, _mvpGpu, _originGpu, RAY2D_GRID_WIDTH,
			RAY2D_GRID_HEIGHT, RAY2D_GRID_DEPTH);

		// Check for errors and synchronize
		gpuErrchk(cudaPeekAtLastError());

		// Unmap pbo
		gpuErrchk(cudaGraphicsUnmapResources(1, &_glFb, 0));

		// Render pbo to screen
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
		glDrawPixels(_width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		//glBegin(GL_QUADS);
		//glVertex2f(-1.0f, -1.0f);
		//glVertex2f(1.0f, -1.0f);
		//glVertex2f(1.0f, 1.0f);
		//glVertex2f(-1.0f, 1.0f);
		//glEnd();
	}

	void CudaTest::genGrid()
	{
		// Initialise grid with random data
		long long length = RAY2D_GRID_WIDTH * RAY2D_GRID_HEIGHT * RAY2D_GRID_DEPTH;
		
		for (long long i = 0; i < length; ++i)
		{
			((int*)_grid)[i] = rand() % 10 < 3;
		}
	}

	void CudaTest::mouse_callback(GLFWwindow* window, int button,
		int action, int mods)
	{
		// Get class instance
		CudaTest* cudatest = (CudaTest*)glfwGetWindowUserPointer(window);
	}

	void CudaTest::mouse_move_callback(GLFWwindow* window, double x, double y)
	{
		// Get class instance
		CudaTest* cudatest = (CudaTest*)glfwGetWindowUserPointer(window);
	}

	void CudaTest::key_callback(GLFWwindow* window, int key,
		int scancode, int action, int mods)
	{
		// Do default action (exit on esc)
		_default_key_callback(window, key, scancode, action, mods);
		
		// Get class instance
		CudaTest* cudatest = (CudaTest*)glfwGetWindowUserPointer(window);
	}
}
