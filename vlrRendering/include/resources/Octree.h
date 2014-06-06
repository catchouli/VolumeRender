#ifndef VLR_RENDERING_OCTREE
#define VLR_RENDERING_OCTREE

#include "Mesh.h"
#include "../rendering/child_desc.h"

#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdint.h>

#include <functional>
#include <stdint.h>

namespace vlr
{
	namespace rendering
	{
		struct child_desc_builder
		{
			uint32_t child_id : 24;
			uint32_t child_mask : 8;
			uint32_t non_leaf_mask : 8;
			
			raw_attachment_uncompressed shading_attributes[8];
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
			uintptr_t ptr;
			uintptr_t rel;

			bool far;
			uintptr_t far_ptr;
		};

		typedef std::function<bool(glm::vec3, glm::vec3, raw_attachment_uncompressed& shading_attributes)> point_test_func;
		
		int32_t genOctree(int32_t** ret, int32_t max_depth,
			point_test_func& test_point_func,
			const glm::vec3& min, const glm::vec3& max);

		bool meshAABBIntersect(Mesh* mesh, glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes);
		__device__ __host__ bool cubeSphereSurfaceIntersection(glm::vec3 centre, float half_size, glm::vec3 pos, float r);
		__device__ __host__ bool boxSphereIntersection(glm::vec3 min, glm::vec3 max, glm::vec3 pos, float r);
		
		int32_t genOctreeSphere(int32_t** ret, int32_t resolution, glm::vec3 pos, float radius);
		int32_t genOctreeMesh(int32_t** ret, int32_t resolution, Mesh* mesh);
	}
}

#endif /* VLR_RENDERING_OCTREE */
