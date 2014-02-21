#include "rendering/RocketRenderer.h"

#include <GL/glew.h>

#include "resources/Image.h"

namespace vlr
{
	RocketRenderer::RocketRenderer()
	{

	}

	void RocketRenderer::RenderGeometry(Rocket::Core::Vertex* vertices, int unused,
		int* indices, int num_indices, Rocket::Core::TextureHandle texture,
		const Rocket::Core::Vector2f& translation)
	{
		// Set up view matrix
		glMatrixMode(GL_MODELVIEW);

		// Push matrix
		glPushMatrix();

		// Apply translation
		glTranslatef(translation.x, translation.y, 0);
		
		// Set vertex pointer
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(2, GL_FLOAT, sizeof(Rocket::Core::Vertex), &(vertices[0].position));
		
		// Set colour pointer
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Rocket::Core::Vertex), &(vertices[0].colour));

		// Enable blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// Bind texture & enable texture mapping
		if (!texture)
		{
			glDisable(GL_TEXTURE_2D);
			glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		}
		else
		{
			glEnable(GL_TEXTURE_2D);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexCoordPointer(2, GL_FLOAT, sizeof(Rocket::Core::Vertex), &vertices[0].tex_coord);
		}

		// Draw
		glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, indices);

		// Restore matrix
		glPopMatrix();
	}

	void RocketRenderer::EnableScissorRegion(bool enable)
	{
		if (enable)
			glEnable(GL_SCISSOR_TEST);
		else
			glDisable(GL_SCISSOR_TEST);
	}

	void RocketRenderer::SetScissorRegion(int x, int y, int width, int height)
	{
		glScissor(x, _height - (y + height), width, height);
	}

	bool RocketRenderer::LoadTexture(
		Rocket::Core::TextureHandle& texture_handle,
		Rocket::Core::Vector2i& texture_dimensions,
		const Rocket::Core::String& source)
	{
		common::Image image;

		if (!image.load(source.CString()))
			return false;

		texture_handle = image.genGlTexture();
		texture_dimensions.x = image.getWidth();
		texture_dimensions.y = image.getHeight();

		return true;
	}

	bool RocketRenderer::GenerateTexture(
		Rocket::Core::TextureHandle& texture_handle,
		const Rocket::Core::byte* source,
		const Rocket::Core::Vector2i& source_dimensions)
	{
		common::Image image;

		// Set image pointer
		image.setPointer((void*)source, source_dimensions.x, source_dimensions.y);

		// Generate texture
		texture_handle = image.genGlTexture();

		// Clear image pointer so the destructor doesn't try to free it
		image.setPointer(nullptr, 0, 0);

		return true;
	}

	void RocketRenderer::ReleaseTexture(Rocket::Core::TextureHandle texture_handle)
	{
		GLuint handle = texture_handle;
		glDeleteTextures(1, &handle);
	}
}
