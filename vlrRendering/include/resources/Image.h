
#ifndef VLR_RENDERING_IMAGE
#define VLR_RENDERING_IMAGE

#include <string.h>
#include <GL/glew.h>
#include <stdint.h>

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

			inline HOST_DEVICE_FUNC int32_t getWidth() const;
			inline HOST_DEVICE_FUNC int32_t getHeight() const;

			inline HOST_DEVICE_FUNC void* getPointer();

			inline void setPointer(void* pointer, int32_t width, int32_t height);

		private:
			void* _pixels;
			int32_t _width, _height;
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

				_pixels = new int32_t[_width * _height];

				memcpy(_pixels, other._pixels,
					sizeof(int32_t) * _width * _height);
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

		int32_t Image::getWidth() const
		{
			return _width;
		}

		int32_t Image::getHeight() const
		{
			return _height;
		}

		void* Image::getPointer()
		{
			return _pixels;
		}

		void Image::setPointer(void* pointer, int32_t width, int32_t height)
		{
			_pixels = pointer;
			_width = width;
			_height = height;
		}
	}
}

#endif /* VLR_RENDERING_IMAGE */
