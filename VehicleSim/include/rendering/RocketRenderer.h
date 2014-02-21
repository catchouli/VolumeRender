#ifndef VEHICLESIM_ROCKETRENDERER
#define VEHICLESIM_ROCKETRENDERER

#include <Rocket/Core.h>

#include "resources/Image.h"

namespace vlr
{
	class RocketRenderer
		: public Rocket::Core::RenderInterface
	{
	public:
		RocketRenderer();

		void RenderGeometry(Rocket::Core::Vertex* vertices, int num_vertices,
						int* indices, int num_indices, Rocket::Core::TextureHandle texture,
						const Rocket::Core::Vector2f& translation) override;

		void EnableScissorRegion(bool enable) override;
		void SetScissorRegion(int x, int y, int width, int height) override;

		bool LoadTexture(Rocket::Core::TextureHandle& texture_handle,
			Rocket::Core::Vector2i& texture_dimensions,
			const Rocket::Core::String& source) override;

		bool RocketRenderer::GenerateTexture(
			Rocket::Core::TextureHandle& texture_handle,
			const Rocket::Core::byte* source,
			const Rocket::Core::Vector2i& source_dimensions) override;

		void ReleaseTexture(Rocket::Core::TextureHandle texture_handle) override;

		void setSize(int width, int height);

	private:
		int _width, _height;
	};

	inline void RocketRenderer::setSize(int width, int height)
	{
		_width = width;
		_height = height;
	}
}

#endif /* VEHICLESIM_ROCKETRENDERER */
