#ifndef VLR_RENDERING_OCTREE
#define VLR_RENDERING_OCTREE

#include "resources/Mesh.h"

#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <functional>

namespace vlr
{
	namespace rendering
	{
		struct child_desc_builder
		{
			unsigned int child_id : 24;
			unsigned int child_mask : 8;
			unsigned int non_leaf_mask : 8;
			
			glm::vec3 norm;
			glm::vec4 col;
		};

		struct child_desc_builder_block
		{
			// Per block attributes
			unsigned int id : 32;
			unsigned int depth : 8;
			unsigned int count : 8;

			unsigned int parent_id : 24;
			unsigned int parent_block_id : 24;

			// Per child desc attributes
			child_desc_builder child_desc_builder[8];
		};

		struct pointer_desc
		{
			int ptr;
			int rel;

			bool far;
			int far_ptr;
		};

		typedef std::function<bool(glm::vec3, glm::vec3, glm::vec3& normal, glm::vec4& colour)> point_test_func;
		
		int genOctree(int** ret, int max_depth,
			point_test_func& test_point_func,
			const glm::vec3& min, const glm::vec3& max);
		
		int genOctreeSphere(int** ret, int resolution, glm::vec3 pos, float radius);
		int genOctreeMesh(int** ret, int resolution, Mesh* mesh);
	}
}

#endif /* VLR_RENDERING_OCTREE */
