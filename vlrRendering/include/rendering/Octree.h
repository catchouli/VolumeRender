#ifndef VLR_RENDERING_OCTREE
#define VLR_RENDERING_OCTREE

#include "resources/Mesh.h"

#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdint.h>

#include <functional>

namespace vlr
{
	namespace rendering
	{
		struct child_desc_builder
		{
			uint32_t child_id : 24;
			uint32_t child_mask : 8;
			uint32_t non_leaf_mask : 8;
			
			glm::vec3 norm;
			glm::vec4 col;
		};

		struct child_desc_builder_block
		{
			// Per block attributes
			uint32_t id : 32;
			uint32_t depth : 8;
			uint32_t count : 8;

			uint32_t parent_id : 24;
			uint32_t parent_block_id : 24;

			// Per child desc attributes
			child_desc_builder child_desc_builder[8];
		};

		struct pointer_desc
		{
			int32_t ptr;
			int32_t rel;

			bool far;
			int32_t far_ptr;
		};

		typedef std::function<bool(glm::vec3, glm::vec3, glm::vec3& normal, glm::vec4& colour)> point_test_func;
		
		int32_t genOctree(int32_t** ret, int32_t max_depth,
			point_test_func& test_point_func,
			const glm::vec3& min, const glm::vec3& max);
		
		int32_t genOctreeSphere(int32_t** ret, int32_t resolution, glm::vec3 pos, float radius);
		int32_t genOctreeMesh(int32_t** ret, int32_t resolution, Mesh* mesh);
	}
}

#endif /* VLR_RENDERING_OCTREE */
