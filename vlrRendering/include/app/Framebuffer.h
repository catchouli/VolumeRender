#ifndef VLR_RENDERING_FRAMEBUFFER
#define VLR_RENDERING_FRAMEBUFFER

#include <GL/glew.h>

namespace vlr
{
	namespace rendering
	{
		class Framebuffer
		{
		public:
			Framebuffer();
			Framebuffer(int width, int height);

			void createGlTexture();
			int* resize(int width, int height);

			void render() const;

			inline int* getPointer();

			inline int getWidth() const;
			inline int getHeight() const;

		private:
			int _width;
			int _height;
			int* _pointer;

			GLuint _texid;
			GLuint _fboid;

			bool _init;
		};

		inline int* Framebuffer::getPointer()
		{
			return _pointer;
		}

		inline int Framebuffer::getWidth() const
		{
			return _width;
		}

		inline int Framebuffer::getHeight() const
		{
			return _height;
		}
	}
}

#endif /* VLR_RENDERING_FRAMEBUFFER */
