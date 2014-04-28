#ifndef VLR_CHILD_DESC
#define VLR_CHILD_DESC

#include <stdint.h>

namespace vlr
{
	namespace rendering
	{
		// This should be a power of two or who knows what'll happen
		// (really it should be 2... why would you change this!)
		const int32_t child_desc_size_ints = 4;
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
			// Order is important due to endianness
			uint32_t nonleaf_mask : 8;
			uint32_t child_mask : 8;
			uint32_t far : 1;
			uint32_t child_ptr : 15;

			// Contour data
			uint32_t contour_ptr : 24;
			uint32_t contour_mask : 8;
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
	}
}

#endif /* VLR_CHILD_DESC */
