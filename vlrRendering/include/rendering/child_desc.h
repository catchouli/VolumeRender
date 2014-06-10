#ifndef VLR_CHILD_DESC
#define VLR_CHILD_DESC

#include <stdint.h>
#include "../maths/Colour.h"
#include "../maths/Normal.h"

#include <stdio.h>

namespace vlr
{
	namespace rendering
	{
		// This should be a power of two or who knows what'll happen
		// (really it should be 2... why would you change this!)
		const int32_t child_desc_size_ints = 2;
		const int32_t child_desc_size = child_desc_size_ints * sizeof(int32_t);

		struct info_section
		{
			// Raw attachment lookup pointer
			int32_t raw_lookup;

			// Compressed attachment lookup pointer
			int32_t compressed_lookup;
		};

		struct child_desc_word_1
		{
			// Child data
			uint32_t nonleaf_mask : 8;
			uint32_t child_mask : 8;
			uint32_t far : 1;
			uint32_t child_ptr : 15;
		};

		union child_desc
		{
			struct child_desc_word_1;
			struct
			{
				uint32_t word_1 : 32;
				uint32_t word_2 : 32;
			};
			int64_t data : 64;
		};

		struct raw_attachment_lookup
		{
			uint32_t ptr : 32;
			//uint32_t mask : 8;
		};

		struct contour
		{
			int32_t nz : 6;
			int32_t ny : 6;
			int32_t nx : 6;
			int32_t position : 7;
			uint32_t thickness : 7;
		};

		struct raw_attachment_uncompressed
		{
			glm::vec3 normal;
			glm::vec4 colour;
			float reflectivity;
			float refractive_index;
		};

		struct raw_attachment
		{
			uint32_t normal;
			uint32_t colour;
			uint32_t reflectivity : 16;
			uint32_t refractive_index : 16;
		};

		__device__ __host__ inline uint32_t pack_float(float value, int bits, float min, float max)
		{
			float range = max - min;

			float max_value = (float)(1 << (bits-1));

			// Convert to a float between 0 and number of bits max
			value -= min;
			value *= (max_value / range);

			return (int)value;
		}

		__device__ __host__ inline float unpack_float(uint32_t value, int bits, float min, float max)
		{
			float range = max - min;

			float max_value = (float)(1 << (bits-1));

			// Convert back
			float val_float = value * (range / max_value);
			val_float += min;

			return val_float;
		}

		__device__ __host__ inline void pack_raw_attachment(const raw_attachment_uncompressed& uncompressed, raw_attachment& compressed)
		{
			compressed.colour = compressColour(uncompressed.colour);
			compressed.normal = compressNormal(uncompressed.normal);
			compressed.reflectivity = pack_float(uncompressed.reflectivity, 16, 0.0f, 1.0f);
			compressed.refractive_index = pack_float(uncompressed.refractive_index, 16, 1.0f, 10.0f);
		}

		__device__ __host__ inline void unpack_raw_attachment(const raw_attachment& compressed, raw_attachment_uncompressed& uncompressed)
		{
			uncompressed.colour = decompressColour(compressed.colour);
			uncompressed.normal = decompressNormal(compressed.normal);
			uncompressed.reflectivity = unpack_float(compressed.reflectivity, 16, 0.0f, 1.0f);
			uncompressed.refractive_index = unpack_float(compressed.refractive_index, 16, 1.0f, 10.0f);
		}
	}
}

#endif /* VLR_CHILD_DESC */
