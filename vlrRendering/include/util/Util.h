#ifndef VLR_RENDERING_UTIL
#define VLR_RENDERING_UTIL

#include <vector>
#include <string>
#include <sstream>

#include <cuda_runtime_api.h>

#undef min
#undef max

#define CONCAT(f) ((std::stringstream&)(std::stringstream() << f)).str();

namespace vlr
{
	namespace rendering
	{
		inline bool file_exists(const char* filename)
		{
			FILE *file;

			if (file = fopen(filename, "r"))
			{
				fclose(file);
				return true;
			}

			return false;
		}

		inline size_t read_full_file_mode(const char* filename, char** ret, const char* mode)
		{
			FILE* file;
			long file_len;

			// Open file for reading
			file = fopen(filename, mode);

			if (file == NULL)
			{
				fprintf(stderr, "Failed to open file for reading: %s\n", filename);

				*ret = nullptr;
				return 0;
			}

			// Get length of file
			fseek(file, 0, SEEK_END);
			file_len = ftell(file);
			fseek(file, 0, SEEK_SET);

			// Allocate memory
			*ret = (char*)malloc(file_len * sizeof(char));

			// Read file into buffer
			size_t read = fread(*ret, sizeof(char), file_len, file);

			fclose(file);

			return read;
		}

		inline size_t read_full_file(const char* filename, char** ret)
		{
			return read_full_file_mode(filename, ret, "r");
		}

		inline size_t read_full_file_binary(const char* filename, char** ret)
		{
			return read_full_file_mode(filename, ret, "rb");
		}
		inline glm::vec3 closestPointOnTriangle( glm::vec3 triangle[3], const glm::vec3 &sourcePosition )
		{
			glm::vec3 edge0 = triangle[1] - triangle[0];
			glm::vec3 edge1 = triangle[2] - triangle[0];
			glm::vec3 v0 = triangle[0] - sourcePosition;

			auto clamp = [] (float a, float min, float max) { return std::min(std::max(a, min), max); };

			float a = glm::dot(edge0, edge0);
			float b = glm::dot(edge0, edge1);
			float c = glm::dot(edge1, edge1);
			float d = glm::dot(edge0, v0);
			float e = glm::dot(edge1, v0);

			float det = a*c - b*b;
			float s = b*e - c*d;
			float t = b*d - a*e;

			if ( s + t < det )
			{
				if ( s < 0.f )
				{
					if ( t < 0.f )
					{
						if ( d < 0.f )
						{
							s = clamp( -d/a, 0.f, 1.f );
							t = 0.f;
						}
						else
						{
							s = 0.f;
							t = clamp( -e/c, 0.f, 1.f );
						}
					}
					else
					{
						s = 0.f;
						t = clamp( -e/c, 0.f, 1.f );
					}
				}
				else if ( t < 0.f )
				{
					s = clamp( -d/a, 0.f, 1.f );
					t = 0.f;
				}
				else
				{
					float invDet = 1.f / det;
					s *= invDet;
					t *= invDet;
				}
			}
			else
			{
				if ( s < 0.f )
				{
					float tmp0 = b+d;
					float tmp1 = c+e;
					if ( tmp1 > tmp0 )
					{
						float numer = tmp1 - tmp0;
						float denom = a-2*b+c;
						s = clamp( numer/denom, 0.f, 1.f );
						t = 1-s;
					}
					else
					{
						t = clamp( -e/c, 0.f, 1.f );
						s = 0.f;
					}
				}
				else if ( t < 0.f )
				{
					if ( a+d > b+e )
					{
						float numer = c+e-b-d;
						float denom = a-2*b+c;
						s = clamp( numer/denom, 0.f, 1.f );
						t = 1-s;
					}
					else
					{
						s = clamp( -e/c, 0.f, 1.f );
						t = 0.f;
					}
				}
				else
				{
					float numer = c+e-b-d;
					float denom = a-2*b+c;
					s = clamp( numer/denom, 0.f, 1.f );
					t = 1.f - s;
				}
			}

			return triangle[0] + s * edge0 + t * edge1;
		}
	}
}

#endif /* VLR_RENDERING_UTIL */
