#include "app/Framebuffer.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

namespace vlr
{
	namespace common
	{
		Framebuffer::Framebuffer()
			: _width(0), _height(0), _pointer(nullptr), _init (false)
		{

		}

		Framebuffer::Framebuffer(int width, int height)
			: _width(width), _height(height), _pointer(nullptr), _init(true)
		{
			// Create opengl texture
			createGlTexture();

			// Initialise framebuffer
			resize(width, height);
		}

		void Framebuffer::createGlTexture()
		{
			// Create texture
			glGenTextures(1, &_texid);
			glBindTexture(GL_TEXTURE_2D, _texid);

			// Create FBO
			glGenFramebuffers(1, &_fboid);
			
			// Bind FBO
			glBindFramebuffer(GL_READ_FRAMEBUFFER, _fboid);

			// Bind texture to FBO
			glFramebufferTexture(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				_texid, 0);
			
			int i = glGetError();

			// Bind default framebuffer
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		}

		int* Framebuffer::resize(int width, int height)
		{
			if (!_init)
			{
				createGlTexture();
				_init = true;
			}

			// Store width and height
			_width = width;
			_height = height;
			
			// Reallocate memory
			_pointer = (int*)realloc(_pointer, width * height * sizeof(int));

			// Return new pointer
			return _pointer;
		}

		void Framebuffer::render() const
		{
			if (_pointer != nullptr)
			{
				// Bind texture
				glBindTexture(GL_TEXTURE_2D, _texid);

				// Update texture
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, _pointer);

				// Bind read framebuffer for texture
				glBindFramebuffer(GL_READ_FRAMEBUFFER, _fboid);

				// Copy texture to framebuffer
				glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

				// Bind default framebuffer
				glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

				// Unbind texture
				glBindTexture(GL_TEXTURE_2D, 0);
			}
		}
	}
}
