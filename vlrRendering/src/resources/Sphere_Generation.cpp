#include "resources/Octree.h"

#include "util/Util.h"
#include "util/CUDAUtil.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vlr
{
	namespace rendering
	{
		__device__ __host__ float sqr(float a)
		{
			return a * a;
		}

		// Check whether a box intersects a sphere by calculating the nearest point on
		// any face of the cube to the sphere's centre and then comparing it to r^2
		__device__ __host__ bool boxSphereIntersection(glm::vec3 min, glm::vec3 max, glm::vec3 pos, float r)
		{
			float r_squared = sqr(r);
			float nearest = 0;

			for (int32_t i = 0; i < 3; i++)
			{
				if(pos[i] < min[i])
					nearest += sqr(pos[i] - min[i]);
				else if(pos[i] > max[i])
					nearest += sqr(pos[i] - max[i]);
			}

			return nearest <= r_squared;
		}

		// Check whether a box intersects surface of sphere
		__device__ __host__ bool cubeSphereSurfaceIntersection(glm::vec3 centre, float half_size, glm::vec3 pos, float r)
		{
			glm::vec3 min = centre - glm::vec3(half_size, half_size, half_size);
			glm::vec3 max = centre + glm::vec3(half_size, half_size, half_size);

			bool intersectsSphere = boxSphereIntersection(min, max, pos, r);

			if (!intersectsSphere)
				return false;

			// Check if cube is fully enclosed in sphere
			// Check each vertex, if any is outside the sphere, this box is not fully enclosed
			for (float x = 0; x <= 1; ++x)
			{
				for (float y = 0; y <= 1; ++y)
				{
					for (float z = 0; z <= 1; ++z)
					{
						// Check if this vertex is outside of the sphere
						int32_t xsign = x > 0 ? 1 : -1;
						int32_t ysign = y > 0 ? 1 : -1;
						int32_t zsign = z > 0 ? 1 : -1;

						glm::vec3 vertex = centre + glm::vec3(xsign * half_size,
							ysign * half_size, zsign * half_size);

						// Get square distance between sphere centre and vertex
						glm::vec3 diff = vertex - pos;
						float sqrDist = glm::dot(diff, diff);

						// If this vertex is outside the sphere, we good
						if (sqrDist > sqr(r))
							return true;
					}
				}
			}

			// All vertices inside sphere
			return false;
		}

		int32_t genOctreeSphere(int32_t** ret, int32_t resolution, glm::vec3 pos, float radius)
		{
			auto test_func = [pos, radius] (glm::vec3 min, glm::vec3 max, glm::vec3& outnormal, glm::vec4& outcolour)
			{
				glm::vec3 half_size = 0.5f * (max - min);
				glm::vec3 centre = min + half_size;
				glm::vec3 normal = glm::normalize(0.5f*pos - centre);
				glm::vec3 normal2 = glm::normalize(pos+0.5f*pos - centre);

				//outnormal.x = std::min((uint32_t)(normal.x * 127.5f + 127.5f), 255u);
				//outnormal.y = std::min((uint32_t)(normal.y * 127.5f + 127.5f), 255u);
				//outnormal.z = std::min((uint32_t)(normal.z * 127.5f + 127.5f), 255u);

				//uint32_t white = (uint32_t)-1;
				//outcolour = *(uchar4*)&white;

				outnormal = normal;
				outcolour = glm::vec4(1.0f);

				// Calculate contours
				glm::vec3 posOnSphere = pos + outnormal * radius;

				glm::vec3 plane1pos = posOnSphere;

				//glm::mat4 lookatmatrix = glm::gtc::matrix_transform::lookAt(pos, plane1pos, glm::vec3(0, 1, 0));

				//glm::vec3 intersection(sqrt(sqr(radius) - 

				return cubeSphereSurfaceIntersection(centre, half_size.x, pos, radius);

				//bool test1 = cubeSphereSurfaceIntersection(centre, half_size.x, pos/2.0f, radius);
				//bool test2 = cubeSphereSurfaceIntersection(centre, half_size.x, pos + pos/2.0f, radius);
				//
				//if (test1)
				//{
				//	outnormal = normal;
				//	outcolour = glm::vec4(1.0f, 0, 0, 0);
				//}
				//if (test2)
				//{
				//	outnormal = normal2;
				//	outcolour = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
				//}

				////return boxSphereIntersection(min, max, pos, radius);
				//return test1 || test2;
			};
			
			point_test_func func(test_func);

			glm::vec3 min;
			glm::vec3 max(1, 1, 1);

			return genOctree(ret, resolution, func, min, max);
		}
	}
}