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
		__global__ void boxMeshIntersect(SubMesh* subMesh, int32_t** textures,
			glm::vec3 min, glm::vec3 max, bool* hit, uchar4* outnormal, uchar4* outcolour)
		{
			int32_t tri_index = blockIdx.x * blockDim.x + threadIdx.x;

			int32_t initial_index = tri_index * 3;

			if (initial_index+2 >= subMesh->_indexCount)
				return;

			int32_t* indices = &subMesh->_indices[initial_index];
			
			glm::vec3 half_size = 0.5f * (max - min);
			glm::vec3 centre = min + half_size;
			
			Vertex* v1 = &subMesh->_vertices[indices[0]];
			Vertex* v2 = &subMesh->_vertices[indices[1]];
			Vertex* v3 = &subMesh->_vertices[indices[2]];
			
			glm::vec3 v1p = v1->_pos;
			glm::vec3 v2p = v2->_pos;
			glm::vec3 v3p = v3->_pos;
			
			float tri[3][3] =
			{
				{ v1p.x, v1p.y, v1p.z},
				{ v2p.x, v2p.y, v2p.z},
				{ v3p.x, v3p.y, v3p.z}
			};

			if (triBoxOverlap((float*)&centre, (float*)&half_size, tri))
				*hit = true;

			//glm::vec3 pos(0.5f, 0.5f, 0.5f);
			//float radius = 0.5f;

			//glm::vec3 centre = min + 0.5f * (max - min);
			//glm::vec3 normal = glm::normalize(centre - pos);
			////outnormal->x = std::min((uint32_t)(normal.x * 127.5f + 127.5f), 255u);
			////outnormal->y = std::min((uint32_t)(normal.y * 127.5f + 127.5f), 255u);
			////outnormal->z = std::min((uint32_t)(normal.z * 127.5f + 127.5f), 255u);

			//uint32_t white = (uint32_t)-1;
			////*outcolour = *(uchar4*)&white;

			//if (boxSphereIntersection(min, max, pos, radius))
			//	*hit = true;
		}

		int32_t genOctreeMesh(int32_t** ret, int32_t resolution, Mesh* mesh)
		{
			// Allocate memory on the gpu for mesh
			int32_t subMeshCount = mesh->getSubMeshCount();
			SubMesh* gpuSubmeshes;

			gpuErrchk(cudaMalloc((void**)&gpuSubmeshes, subMeshCount * sizeof(SubMesh)));

			// Copy each submesh to gpu
			for (int32_t i = 0; i < subMeshCount; ++i)
			{
				SubMesh gpuSubMesh;

				const SubMesh* originalSubMesh = mesh->getSubMesh(i);

				// Allocate memory for vertices
				gpuSubMesh._vertexCount = originalSubMesh->_vertexCount;
				gpuErrchk(cudaMalloc((void**)&gpuSubMesh._vertices,
					gpuSubMesh._vertexCount * sizeof(Vertex)));

				// Allocate memory for indices
				gpuSubMesh._indexCount = originalSubMesh->_indexCount;
				gpuErrchk(cudaMalloc((void**)&gpuSubMesh._indices,
					gpuSubMesh._indexCount * sizeof(int32_t)));

				// Copy indices and vertices
				gpuErrchk(cudaMemcpy(gpuSubMesh._indices, originalSubMesh->_indices,
					originalSubMesh->_indexCount * sizeof(int32_t), cudaMemcpyHostToDevice));

				gpuErrchk(cudaMemcpy(gpuSubMesh._vertices, originalSubMesh->_vertices,
					originalSubMesh->_vertexCount * sizeof(Vertex), cudaMemcpyHostToDevice));

				// Copy SubMesh
				gpuErrchk(cudaMemcpy(gpuSubmeshes, &gpuSubMesh,
					sizeof(SubMesh), cudaMemcpyHostToDevice));

				// Clear gpu pointers from submesh so we don't cause delete[] on an invalid pointer
				gpuSubMesh._indices = nullptr;
				gpuSubMesh._vertices = nullptr;
			}

			auto test_func = [&] (glm::vec3 min, glm::vec3 max, glm::vec3& outnormal, glm::vec4& outcolour)
			{
				//// Test hit for each submesh
				//for (int32_t i = 0; i < subMeshCount; ++i)
				//{
				//	const SubMesh* curMesh = mesh->getSubMesh(i);
				//	SubMesh* gpuCurMesh = gpuSubmeshes + i;

				//	bool hit = false;

				//	bool* gpuHit;
				//	gpuErrchk(cudaMalloc((void**)&gpuHit, sizeof(bool)));
				//	gpuErrchk(cudaMemcpy(gpuHit, &hit, sizeof(bool), cudaMemcpyHostToDevice));

				//	// Run test kernel
				//	int32_t triCount = curMesh->_indexCount / 3;

				//	int32_t blocks = 256;
				//	int32_t threads = triCount / blocks + 1;

				//	boxMeshIntersect<<<blocks, threads>>>(gpuCurMesh, nullptr, min, max, gpuHit, nullptr, nullptr);

				//	gpuErrchk(cudaDeviceSynchronize());

				//	gpuErrchk(cudaMemcpy(&hit, gpuHit, sizeof(bool), cudaMemcpyDeviceToHost));

				//	if (hit)
				//		return true;
				//}
				//
				//return false;

				// Old vers.
				glm::vec3 size = max - min;
				glm::vec3 half_size = 0.5f * size;
				glm::vec3 centre = min + half_size;

				for (int32_t i = 0; i < mesh->getSubMeshCount(); ++i)
				{
					const SubMesh* sub_mesh = mesh->getSubMesh(i);

					for (int32_t j = 0; j < sub_mesh->_indexCount; j += 3)
					{
						//ozcollide::Vec3f tri[3];

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
						
						//tri[0].x = v1p.x;
						//tri[0].y = v1p.y;
						//tri[0].z = v1p.z;

						//tri[1].x = v2p.x;
						//tri[1].y = v2p.y;
						//tri[1].z = v2p.z;

						//tri[2].x = v3p.x;
						//tri[2].y = v3p.y;
						//tri[2].z = v3p.z;

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

							// Get texture
							if (!mesh->hasTextures())
							{
								outcolour = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
							}
							else
							{
								Image* images = mesh->getStoredTextures();
								Image* image = &images[sub_mesh->_materialIndex];
							
								// Interpolate uvs on surface
								float tex_u = u * v1->_texCoord.x + v * v2->_texCoord.x + w * v3->_texCoord.x;
								float tex_v = u * v1->_texCoord.y + v * v2->_texCoord.y + w * v3->_texCoord.y;
							
								int32_t x = (int32_t)(tex_u * image->getWidth());
								int32_t y = (int32_t)(tex_v * image->getHeight());

								// Get colour from texture
								int32_t* ptr = (int32_t*)image->getPointer() + y * image->getWidth() + x;
							
								uchar4 col = *(uchar4*)ptr;

								//outnormal.x = std::min((uint32_t)(normal.x * 127.5f + 127.5f), 255u);
								//outnormal.y = std::min((uint32_t)(normal.y * 127.5f + 127.5f), 255u);
								//outnormal.z = std::min((uint32_t)(normal.z * 127.5f + 127.5f), 255u);
								//
								//outcolour = *(uchar4*)&col;

								outnormal = normal;
								outcolour = glm::vec4(col.x, col.y, col.z, 0.0f) / 255.0f;

								// Convert from BGR to RGB
								float temp = outcolour.x;
								outcolour.x = outcolour.z;
								outcolour.z = temp;
							}

							return true;
						}
					}
				}

				return false;
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
