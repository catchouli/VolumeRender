#include "resources/Mesh.h"

#include "resources/Image.h"
#include "util/Util.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <gl/glew.h>

namespace vlr
{
	namespace rendering
	{
		Mesh::Mesh(bool storeTextures, aiPostProcessSteps normalMode)
			: _loaded(false), _subMeshCount(0),
			_subMeshes(nullptr), _textures(nullptr),
			_storeTextures(storeTextures), _images(nullptr),
			_normalMode(normalMode)
		{

		}

		Mesh::Mesh(const char* filename, bool storeTextures, aiPostProcessSteps normalMode)
			: _loaded(false), _subMeshCount(0),
			_subMeshes(nullptr), _textures(nullptr),
			_storeTextures(storeTextures), _images(nullptr),
			_normalMode(normalMode)
		{
			load(filename);
		}

		Mesh::Mesh(const Mesh& other)
			: _loaded(other._loaded), _subMeshCount(other._subMeshCount),
			_subMeshes(nullptr), _textures(nullptr),
			_storeTextures(other._storeTextures), _images(nullptr),
			_min(other._min), _max(other._max), _textureCount(other._textureCount),
			_normalMode(other._normalMode)
		{
			if (!_loaded)
				return;

			// Copy submeshes
			_subMeshes = new SubMesh[_subMeshCount];

			for (size_t i = 0; i < _subMeshCount; ++i)
			{
				// Copy vertices and indices
				SubMesh& currentSubMesh = _subMeshes[i];
				SubMesh& originalSubMesh = other._subMeshes[i];
				
				int32_t j = originalSubMesh._indexCount;
				currentSubMesh._indexCount = j;
				currentSubMesh._vertexCount = originalSubMesh._vertexCount;
				currentSubMesh._materialIndex = originalSubMesh._materialIndex;
				
				currentSubMesh._vertices = new Vertex[currentSubMesh._vertexCount];
				currentSubMesh._indices = new int32_t[currentSubMesh._indexCount];

				// Copy indices
				memcpy(currentSubMesh._indices, originalSubMesh._indices,
					currentSubMesh._indexCount * sizeof(int32_t));

				// Copy vertices
				for (int32_t i = 0; i < currentSubMesh._vertexCount; ++i)
				{
					Vertex& currentVertex = currentSubMesh._vertices[i];
					Vertex& originalVertex = originalSubMesh._vertices[i];
					
					currentVertex._pos = originalVertex._pos;
					currentVertex._normal = originalVertex._normal;
					currentVertex._texCoord = originalVertex._texCoord;
				}

				// Copy materials
				if (other._textures != nullptr)
				{
					_textures = new GLuint[_textureCount];
					memcpy(_textures, other._textures,
						sizeof(GLuint) * _textureCount);
				}

				// Copy images if stored
				if (other._images != nullptr)
				{
					_images = new Image[_textureCount];

					for (size_t i = 0; i < _textureCount; ++i)
					{
						_images[i] = other._images[i];
					}
				}
			}
		}

		Mesh::~Mesh()
		{
			unload();

			_test = false;
		}

		void Mesh::unload()
		{
			delete[] _images;
			delete[] _subMeshes;
			delete[] _textures;

			_subMeshes = nullptr;
			_textures = nullptr;
			_images = nullptr;

			_subMeshCount = 0;

			_loaded = false;
		}

		Mesh& Mesh::operator=(const Mesh& other)
		{
			Mesh temp(other);
			
			test::swap(_loaded, temp._loaded);
			test::swap(_storeTextures, temp._storeTextures);

			test::swap(_subMeshes, temp._subMeshes);
			test::swap(_subMeshCount, temp._subMeshCount);

			test::swap(_textureCount, temp._textureCount);
			
			test::swap(_textures, temp._textures);
			test::swap(_images, temp._images);
			
			test::swap(_min, temp._min);
			test::swap(_max, temp._max);

			return *this;
		}

		void Mesh::render()
		{
			// For each submesh
			for (size_t i = 0; i < _subMeshCount; ++i)
			{
				SubMesh* currentMesh = &_subMeshes[i];
				
				glEnableClientState(GL_VERTEX_ARRAY);
				glEnableClientState(GL_NORMAL_ARRAY);

				if (hasTextures())
				{
					glEnable(GL_TEXTURE_2D);

					glEnableClientState(GL_TEXTURE_COORD_ARRAY);

					glTexCoordPointer(2, GL_FLOAT, sizeof(Vertex), &currentMesh->_vertices->_texCoord);

					glBindTexture(GL_TEXTURE_2D, _textures[currentMesh->_materialIndex]);
				}

				glVertexPointer(3, GL_FLOAT, sizeof(Vertex), currentMesh->_vertices);
				glNormalPointer(GL_FLOAT, sizeof(Vertex), &currentMesh->_vertices->_normal);

				glDrawElements(GL_TRIANGLES, currentMesh->_indexCount, GL_UNSIGNED_INT, currentMesh->_indices);
				
				glDisableClientState(GL_VERTEX_ARRAY);
				glDisableClientState(GL_NORMAL_ARRAY);
				glDisableClientState(GL_TEXTURE_COORD_ARRAY);
			}
		}

		void Mesh::transform(const glm::mat4& matrix)
		{
			// Transform all points
			for (size_t i = 0; i < _subMeshCount; ++i)
			{
				SubMesh& subMesh = _subMeshes[i];

				for (int32_t j = 0; j < subMesh._vertexCount; ++j)
				{
					glm::vec4 vert = glm::vec4(subMesh._vertices[j]._pos, 1.0f);
					glm::vec4 norm = glm::vec4(subMesh._vertices[j]._normal, 0);
					
					vert = matrix * vert;
					norm = matrix * norm;

					subMesh._vertices[j]._pos = glm::vec3(vert);
					subMesh._vertices[j]._normal = glm::vec3(norm);
				}
			}

			// Recalculate min/max
			calcMinMax();
		}

		void Mesh::calcMinMax()
		{
			// Calculate min/max
			_min = _subMeshes[0]._vertices[0]._pos;
			_max = _subMeshes[0]._vertices[0]._pos;

			for (size_t i = 0; i < _subMeshCount; ++i)
			{
				SubMesh& subMesh = _subMeshes[i];

				for (int32_t j = 0; j < subMesh._vertexCount; ++j)
				{
					glm::vec3 vert = subMesh._vertices[j]._pos;
					
					if (vert.x < _min.x)
						_min.x = vert.x;
					if (vert.y < _min.y)
						_min.y = vert.y;
					if (vert.z < _min.z)
						_min.z = vert.z;

					if (vert.x > _max.x)
						_max.x = vert.x;
					if (vert.y > _max.y)
						_max.y = vert.y;
					if (vert.z > _max.z)
						_max.z = vert.z;
				}
			}
		}

		bool Mesh::load(const char* filename)
		{
			Assimp::Importer importer;

			const aiScene* scene;

			// Unload previously loaded mesh
			unload();

			// Check file exists
			if (!file_exists(filename))
			{
				fprintf(stderr, "Could not open model file %s\n", filename);
				return false;
			}

			// Attempt to load mesh
			scene = importer.ReadFile(filename,
				aiProcess_Triangulate | _normalMode | aiProcess_FlipUVs);

			if (!scene)
			{
				fprintf(stderr, "Failed to parse model file %s\n", filename);
				return false;
			}

			// Load materials
			if (scene->HasMaterials())
			{
				_textureCount = scene->mNumMaterials;
				_textures = new GLuint[_textureCount];
				_images = new Image[_textureCount];

				for (uint32_t i = 0; i < scene->mNumMaterials; ++i)
				{
					// Load material
					const aiMaterial* material = scene->mMaterials[i];

					if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
					{
						aiString path;

						if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
						{
							if (!_images[i].load(path.C_Str()))
							{
								fprintf(stderr, "Failed to load texture: %s\n", path);
							}
							else
							{
								_textures[i] = _images[i].genGlTexture();
							}
						}
						else
						{
							fprintf(stderr, "No texture available for material\n");
						}
					}
					else
					{
						fprintf(stderr, "No diffuse texture for material\n");
					}
				}

				// Clean up images
				if (!_storeTextures)
				{
					delete[] _images;
					_images = nullptr;
				}
			}

			// Load meshes from assimp scene
			// Allocate memory
			_subMeshCount = scene->mNumMeshes;
			_subMeshes = new SubMesh[_subMeshCount];

			// Initialise meshes
			for (size_t i = 0; i < _subMeshCount; ++i)
			{
				// Get assimp mesh
				const aiMesh* mesh = scene->mMeshes[i];
												
				// Get current submesh
				SubMesh* currentMesh = &_subMeshes[i];

				// Set mesh texture
				currentMesh->_materialIndex = mesh->mMaterialIndex;

				// Allocate memory
				currentMesh->_indexCount = mesh->mNumFaces * 3;
				currentMesh->_vertexCount = mesh->mNumVertices;

				currentMesh->_indices = new int32_t[currentMesh->_indexCount];
				currentMesh->_vertices = new Vertex[currentMesh->_vertexCount];

				// Load vertices
				const aiVector3D ZERO = aiVector3D(0, 0, 0);
				for (uint32_t j = 0; j < mesh->mNumVertices; ++j)
				{
					Vertex& vertex = currentMesh->_vertices[j];
					
					// Load vertex data
					const aiVector3D* pos = &mesh->mVertices[j];
					const aiVector3D* normal = &mesh->mNormals[j];
					const aiVector3D* uv = &ZERO;

					if (mesh->HasTextureCoords((unsigned int)i))
						uv = &mesh->mTextureCoords[0][j];
					
					if (pos->x < _min.x)
						_min.x = pos->x;
					if (pos->y < _min.y)
						_min.y = pos->y;
					if (pos->z < _min.z)
						_min.z = pos->z;
					
					if (pos->x > _max.x)
						_max.x = pos->x;
					if (pos->y > _max.y)
						_max.y = pos->y;
					if (pos->z > _max.z)
						_max.z = pos->z;

					// Convert to vlr format
					vertex._pos = glm::vec3(pos->x, pos->y, pos->z);
					vertex._normal = -1.0f * glm::vec3(normal->x, normal->y, normal->z);
					vertex._texCoord = glm::vec2(uv->x, uv->y);
				}

				// Load indicies
				for (uint32_t j = 0; j < mesh->mNumFaces; ++j)
				{
					assert(mesh->mFaces[j].mNumIndices == 3);
					
					currentMesh->_indices[j * 3 + 0] = mesh->mFaces[j].mIndices[2];
					currentMesh->_indices[j * 3 + 1] = mesh->mFaces[j].mIndices[1];
					currentMesh->_indices[j * 3 + 2] = mesh->mFaces[j].mIndices[0];
				}
			}

			_loaded = true;

			calcMinMax();

			return true;
		}
	}
}
