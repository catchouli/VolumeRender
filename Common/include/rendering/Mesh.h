#ifndef VLR_COMMON_MESH
#define VLR_COMMON_MESH

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

namespace vlr
{
	namespace common
	{
		class Mesh
		{
		public:
			Mesh();
			~Mesh();

			void createCube();

			void cut(Mesh* left, Mesh* right, const glm::vec3& planePoint, const glm::vec3& planeNormal);

			void render();

			glm::vec3* vertices;
			int* indices;
			glm::vec3* vertexNormals;
			glm::vec2* texCoords;
	
			int vertexCount;
			int indexCount;
		};
	}
}

#endif /* VLR_COMMON_MESH */