#ifndef VLR_RENDERING_FRAMEBUFFER
#define VLR_RENDERING_FRAMEBUFFER

#include <GL/glew.h>
#include <stdint.h>

namespace vlr
{
	namespace rendering
	{
		class Framebuffer
		{
		public:
			Framebuffer();
			Framebuffer(int32_t width, int32_t height);

			void createGlTexture();
			int32_t* resize(int32_t width, int32_t height);

			void render() const;

			inline int32_t* getPointer();

			inline int32_t getWidth() const;
			inline int32_t getHeight() const;

		private:
			int32_t _width;
			int32_t _height;
			int32_t* _pointer;

			GLuint _texid;
			GLuint _fboid;

			bool _init;
		};

		inline int32_t* Framebuffer::getPointer()
		{
			return _pointer;
		}

		inline int32_t Framebuffer::getWidth() const
		{
			return _width;
		}

		inline int32_t Framebuffer::getHeight() const
		{
			return _height;
		}
	}
}

#endif /* VLR_RENDERING_FRAMEBUFFER */
