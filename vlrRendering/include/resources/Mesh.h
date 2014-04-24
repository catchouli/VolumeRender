#ifndef VLR_RENDERING_MESH_H
#define VLR_RENDERING_MESH_H

#include <stdio.h>
#include <glm/glm.hpp>
#include <gl/glew.h>

#include "util/CUDAUtil.h"
#include "resources/Image.h"

namespace vlr
{
	namespace rendering
	{
		struct Vertex
		{
			glm::vec3 _pos;
			glm::vec3 _normal;
			glm::vec2 _texCoord;
		};

		class SubMesh
		{
		public:
			HOST_DEVICE_FUNC SubMesh()
				: _indices(nullptr), _vertices(nullptr),
				_indexCount(0), _materialIndex((unsigned int)-1)
			{

			}

			HOST_DEVICE_FUNC ~SubMesh()
			{
				delete[] _indices;
				delete[] _vertices;
			}

			int* _indices;
			int _indexCount;
			Vertex* _vertices;
			int _vertexCount;

			unsigned int _materialIndex;
		};

		class Mesh
		{
		public:
			Mesh(bool storeTextures = false);
			Mesh(const char* filename, bool storeTextures = false);
			Mesh(const Mesh&);
			~Mesh();

			Mesh& operator=(const Mesh& other);

			void render();
			
			bool load(const char* filename);
			void unload();

			void transform(const glm::mat4& matrix);

			inline bool isLoaded() const;

			inline bool hasTextures() const;
			
			inline Image* getStoredTextures() const;

			inline int getSubMeshCount() const;
			inline const SubMesh* getSubMesh(int i) const;
			
			inline const glm::vec3* getMin() const;
			inline const glm::vec3* getMax() const;

		private:
			bool _test;

			bool _loaded;
			bool _storeTextures;

			SubMesh* _subMeshes;
			size_t _subMeshCount;

			size_t _textureCount;

			GLuint* _textures;
			Image* _images;

			glm::vec3 _min;
			glm::vec3 _max;
		};

		bool Mesh::isLoaded() const
		{
			return _loaded;
		}

		bool Mesh::hasTextures() const
		{
			return _textures != nullptr;
		}

		Image* Mesh::getStoredTextures() const
		{
			return _images;
		}

		int Mesh::getSubMeshCount() const
		{
			return (int)_subMeshCount;
		}

		const SubMesh* Mesh::getSubMesh(int i) const
		{
			if (i < 0 || i >= (int)_subMeshCount)
			{
				fprintf(stderr, "Attempted to access invalid submesh\n");

				return nullptr;
			}

			return &_subMeshes[i];
		}
		
		const glm::vec3* Mesh::getMin() const
		{
			return &_min;
		}

		const glm::vec3* Mesh::getMax() const
		{
			return &_max;
		}
	}
}

#endif /* VLR_RENDERING_MESH_H */
