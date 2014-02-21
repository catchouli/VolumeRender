#ifndef VLR_COMMON_IMAGE
#define VLR_COMMON_IMAGE

#include <GL/glew.h>

namespace vlr
{
	namespace common
	{
		class Image
		{
		public:
			Image();
			Image(const char* filename);
			~Image();

			bool load(const char* filename);
			void unload();

			GLuint genGlTexture();

			int getWidth() const;
			int getHeight() const;

			void* getPointer();

			void setPointer(void* pointer, int width, int height);

		protected:
			Image(const Image&);

		private:
			void* _pixels;
			int _width, _height;
		};

		inline int Image::getWidth() const
		{
			return _width;
		}

		inline int Image::getHeight() const
		{
			return _height;
		}

		inline void* Image::getPointer()
		{
			return _pixels;
		}

		inline void Image::setPointer(void* pointer, int width, int height)
		{
			_pixels = pointer;
			_width = width;
			_height = height;
		}
	}
}

#endif /* VLR_COMMON_IMAGE */
