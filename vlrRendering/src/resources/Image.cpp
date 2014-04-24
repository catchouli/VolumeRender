#include "resources/Image.h"

#include "util/CUDAUtil.h"

#include <FreeImage.h>
#include <malloc.h>
#include <stdio.h>

#include <algorithm>

namespace vlr
{
	namespace rendering
	{		
		Image::Image(const char* filename)
			: _pixels(nullptr)
		{
			load(filename);
		}
		
		bool Image::load(const char* filename)
		{
			const unsigned int argb_r = 0xFF000000u;
			const unsigned int argb_g = 0x00FF0000u;
			const unsigned int argb_b = 0x0000FF00u;
			const unsigned int argb_a = 0x000000FFu;

			FREE_IMAGE_FORMAT format;
			FIBITMAP* bitmap;
			FIBITMAP* rgbamap;
			int w, h;

			// Unload if image already loaded to protect from memory leak
			unload();

			// Get filetype
			format = FreeImage_GetFileType(filename);

			if (format == FIF_UNKNOWN)
			{
				fprintf(stderr, "Failed to ascertain filetype of %s\n",
					filename);
				return false;
			}

			// Load image
			bitmap = FreeImage_Load(format, filename);

			if (bitmap == nullptr)
			{
				fprintf(stderr, "Failed to load %s\n", filename);
				return false;
			}

			// Get width and height
			w = FreeImage_GetWidth(bitmap);
			h = FreeImage_GetHeight(bitmap);

			// Convert to RGBA if not already
			rgbamap = FreeImage_ConvertTo32Bits(bitmap);
			int scan_width = FreeImage_GetPitch(rgbamap);

			// Make copy
			int size = h * scan_width;
			_pixels = malloc(size);
			FreeImage_ConvertToRawBits((BYTE*)_pixels, bitmap, scan_width, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

			_width = w;
			_height = h;

			// Unload FreeImage bitmaps
			FreeImage_Unload(bitmap);
			FreeImage_Unload(rgbamap);

			return true;
		}

		GLuint Image::genGlTexture()
		{
			if (_pixels == nullptr)
				return -1;

			GLuint texid;

			// Generate texture
			glGenTextures(1, &texid);
			glBindTexture(GL_TEXTURE_2D, texid);

		   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
						   GL_NEAREST);
		   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
						   GL_NEAREST);

			// Upload texture
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height,
				0, GL_BGRA, GL_UNSIGNED_BYTE, _pixels);

			return texid;
		}
	}
}
