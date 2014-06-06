#include "resources/Octree.h"

#include "util/Util.h"
#include "util/CUDAUtil.h"

#include "trianglebox.c"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace vlr
{
	namespace rendering
	{
		bool meshAABBIntersect(Mesh* mesh, glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
		{
			glm::vec3 size = max - min;
			glm::vec3 half_size = 0.5f * size;
			glm::vec3 centre = min + half_size;

			for (int32_t i = 0; i < mesh->getSubMeshCount(); ++i)
			{
				const SubMesh* sub_mesh = mesh->getSubMesh(i);

				for (int32_t j = 0; j < sub_mesh->_indexCount; j += 3)
				{
					Vertex* v1 = &sub_mesh->_vertices[sub_mesh->_indices[j+0]];
					Vertex* v2 = &sub_mesh->_vertices[sub_mesh->_indices[j+1]];
					Vertex* v3 = &sub_mesh->_vertices[sub_mesh->_indices[j+2]];
						
					const glm::vec3& v1p = v1->_pos;
					const glm::vec3& v2p = v2->_pos;
					const glm::vec3& v3p = v3->_pos;

					float tri[3][3] =
					{
						{ v1p.x, v1p.y, v1p.z},
						{ v2p.x, v2p.y, v2p.z},
						{ v3p.x, v3p.y, v3p.z}
					};

					if (triBoxOverlap((float*)&centre, (float*)&half_size, tri))
					{
						glm::vec3 triangle[3] =
						{
							v1p,
							v2p,
							v3p
						};

						// Get closest point on triangle to cube centre
						glm::vec3 point = closestPointOnTriangle(triangle, centre);

						// Calculate barycentric coordinates
						float tri_area = 0.5f * glm::length(glm::cross(v2p - v1p, v3p - v1p));
						float one_over_tri_area = 1.0f / tri_area;
						float u = 0.5f * one_over_tri_area * glm::length(glm::cross(v2p - point, v3p - point));
						float v = 0.5f * one_over_tri_area * glm::length(glm::cross(v1p - point, v3p - point));
						float w = 0.5f * one_over_tri_area * glm::length(glm::cross(v1p - point, v2p - point));
							
						// Interpolate normal on surface
						const glm::vec3& v1_normal = v1->_normal;
						const glm::vec3& v2_normal = v2->_normal;
						const glm::vec3& v3_normal = v3->_normal;

						glm::vec3 normal = u * v1_normal + v * v2_normal + w * v3_normal;
						shading_attributes.normal = normal;

						assert(mesh->hasTextures() == (mesh->getStoredTextures() != nullptr));

						// Get texture
						if (!mesh->hasTextures())
						{
							shading_attributes.colour = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
						}
						else
						{
							Image* images = mesh->getStoredTextures();

							Image* image = &images[sub_mesh->_materialIndex];

							if (image->getPointer() == nullptr)
							{
								shading_attributes.colour = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
							}
							else
							{							
								// Interpolate uvs on surface
								float tex_u = u * v1->_texCoord.x + v * v2->_texCoord.x + w * v3->_texCoord.x;
								float tex_v = u * v1->_texCoord.y + v * v2->_texCoord.y + w * v3->_texCoord.y;
							
								int32_t x = (int32_t)(tex_u * image->getWidth());
								int32_t y = (int32_t)(tex_v * image->getHeight());

								// Get colour from texture
								int32_t* ptr = (int32_t*)image->getPointer() + y * image->getWidth() + x;
							
								uchar4 col = *(uchar4*)ptr;

								shading_attributes.colour = glm::vec4(col.x, col.y, col.z, 255.0f) / 255.0f;

								// Convert from BGR to RGB
								float temp = shading_attributes.colour.r;
								shading_attributes.colour.r = shading_attributes.colour.b;
								shading_attributes.colour.b = temp;
							}
						}

						return true;
					}
				}
			}

			return false;
		}

		int32_t genOctreeMesh(int32_t** ret, int32_t resolution, Mesh* mesh)
		{
			auto test_func = [&] (glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
			{
				return meshAABBIntersect(mesh, min, max, shading_attributes);
			};
			
			point_test_func func(test_func);

			glm::vec3 min = *mesh->getMin();
			glm::vec3 max = *mesh->getMax();

			// Make cube around bounds
			glm::vec3 extents = (max - min) * 0.5f;
			glm::vec3 centre = min + extents;

			float greatest_extent = std::max(std::max(extents.x, extents.y), extents.z);
			
			extents.x = greatest_extent;
			extents.y = greatest_extent;
			extents.z = greatest_extent;
			
			min = centre - extents;
			max = centre + extents;

			int32_t size = genOctree(ret, resolution, func, min, max);

			return size;
		}
	}
}
