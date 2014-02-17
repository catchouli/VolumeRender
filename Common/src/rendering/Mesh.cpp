#include "rendering/Mesh.h"

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <stdio.h>
#include <string>
#include <fstream>

namespace vlr
{
	namespace common
	{
		Mesh::Mesh()
			: vertexCount(0), indexCount(0), vertices(nullptr), indices(nullptr), vertexNormals(nullptr), texCoords(nullptr)
		{

		}

		Mesh::~Mesh()
		{
			delete[] vertices;
			delete[] indices;
			delete[] vertexNormals;
			delete[] texCoords;
		}

		void Mesh::render()
		{
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_NORMAL_ARRAY);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);

			// Set pointers
			glVertexPointer(3, GL_FLOAT, 0, vertices);
			glNormalPointer(GL_FLOAT, 0, vertexNormals);
			glTexCoordPointer(2, GL_FLOAT, 0, texCoords);

			glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, indices);

			// Reset opengl state
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_NORMAL_ARRAY);
			glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		}

		void Mesh::createCube()
		{
			const int VERTEX_COUNT = 8;
			const int INDEX_COUNT = 36;

			static glm::vec3 VERTICES[VERTEX_COUNT] =
			{
				glm::vec3(1.0f, 0.0f, 0.0f),
				glm::vec3(1.0f, 0.0f,  1.0f),
				glm::vec3(0.0f, 0.0f,  1.0f),
				glm::vec3(0.0f, 0.0f, 0.0f),
				glm::vec3(1.0f,  1.0f, 0.0f),
				glm::vec3(1.0f,  1.0f,  1.0f),
				glm::vec3(0.0f,  1.0f,  1.0f),
				glm::vec3(0.0f,  1.0f, 0.0f)
			};

			static glm::vec3 NORMALS[VERTEX_COUNT] =
			{
				glm::vec3(0.666667f, -0.666667f, -0.333333f),
				glm::vec3(0.408248f, -0.408248f,  0.816497f),
				glm::vec3(-0.666667f, -0.666667f,  0.333333f),
				glm::vec3(-0.408248f, -0.408248f, -0.816497f),
				glm::vec3(0.333333f,  0.666667f, -0.666667f),
				glm::vec3(0.816497f,  0.408248f,  0.408248f),
				glm::vec3(-0.333333f,  0.666667f,  0.666667f),
				glm::vec3(-0.816497f,  0.408248f, -0.408248f)
			};

			static int INDICES[INDEX_COUNT] =
			{
				1, 0, 2, 2, 0, 3,
				7, 4, 6, 6, 4, 5,
				4, 0, 5, 5, 0, 1,
				5, 1, 6, 6, 1, 2,
				6, 2, 7, 7, 2, 3,
				0, 4, 3, 3, 4, 7
			};

			static glm::vec2 UVS[VERTEX_COUNT] =
			{
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
				glm::vec2(0.0f, 0.0f),
			};
		
			vertices = new glm::vec3[VERTEX_COUNT];
			memcpy(vertices, VERTICES, VERTEX_COUNT * sizeof(glm::vec3));

			vertexNormals = new glm::vec3[VERTEX_COUNT];
			memcpy(vertexNormals, NORMALS, VERTEX_COUNT * sizeof(glm::vec3));

			indices = new int[INDEX_COUNT];
			memcpy(indices, INDICES, INDEX_COUNT * sizeof(int));

			texCoords = new glm::vec2[VERTEX_COUNT];
			memcpy(UVS, texCoords, VERTEX_COUNT * sizeof(glm::vec2));

			vertexCount = VERTEX_COUNT;
			indexCount = INDEX_COUNT;
		}
	}
}
