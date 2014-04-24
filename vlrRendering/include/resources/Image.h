#ifndef VLR_RENDERING_IMAGE
#define VLR_RENDERING_IMAGE

#include <string.h>
#include <GL/glew.h>

#include "util/CUDAUtil.h"

namespace vlr
{
	namespace rendering
	{
		class Image
		{
		public:
			inline HOST_DEVICE_FUNC Image();
			Image(const char* filename);
			inline HOST_DEVICE_FUNC Image(const Image&);

			inline HOST_DEVICE_FUNC ~Image();

			inline HOST_DEVICE_FUNC Image& operator=(const Image& other);

			bool load(const char* filename);
			inline HOST_DEVICE_FUNC void unload();

			GLuint genGlTexture();

			inline HOST_DEVICE_FUNC int getWidth() const;
			inline HOST_DEVICE_FUNC int getHeight() const;

			inline HOST_DEVICE_FUNC void* getPointer();

			inline void setPointer(void* pointer, int width, int height);

		private:
			void* _pixels;
			int _width, _height;
		};
		
		Image::Image()
			: _pixels(nullptr)
		{

		}
		
		Image::Image(const Image& other)
			: _pixels(nullptr)
		{
			if (_pixels != nullptr)
			{
				_width = other._width;
				_height = other._height;

				_pixels = new int[_width * _height];

				memcpy(_pixels, other._pixels,
					sizeof(int) * _width * _height);
			}
		}
		
		Image::~Image()
		{
			unload();
		}

		void Image::unload()
		{
			if (_pixels != nullptr)
			{
				free(_pixels);
				_pixels = nullptr;
			}
		}

		Image& Image::operator=(const Image& other)
		{
			Image temp(other);
			
			test::swap(_width, temp._width);
			test::swap(_height, temp._height);
			test::swap(_pixels, temp._pixels);

			return *this;
		}

		int Image::getWidth() const
		{
			return _width;
		}

		int Image::getHeight() const
		{
			return _height;
		}

		void* Image::getPointer()
		{
			return _pixels;
		}

		void Image::setPointer(void* pointer, int width, int height)
		{
			_pixels = pointer;
			_width = width;
			_height = height;
		}
	}
}

#endif /* VLR_RENDERING_IMAGE */
