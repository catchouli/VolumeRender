#ifndef VLR_RENDERING_UTIL
#define VLR_RENDERING_UTIL

#include <vector>
#include <string>
#include <dirent.h>

#include <cuda_runtime_api.h>

#undef min
#undef max

namespace vlr
{
	namespace rendering
	{
		inline std::vector<std::string> filesInDir(const char* directory)
		{
			DIR *dir;
			struct dirent *ent;

			std::vector<std::string> ret;

			if ((dir = opendir (directory)) != NULL)
			{
				// Add all files in directory to return vector
				while ((ent = readdir (dir)) != NULL)
				{
					ret.push_back(std::string(ent->d_name));
				}
				
				closedir(dir);
			}
			else
			{
				fprintf(stderr, "Failed to get files in directory %s\n", dir);
			}

			return ret;
		}

		template <typename T>
		__device__ __host__ inline void swap(T& x, T& y)
		{
			T temp = x;
			x = y;
			y = temp;
		}

		template <typename T>
		__device__ inline const T& min(const T& a, const T& b)
		{
			return a < b ? a : b;
		}

		template <typename T>
		__device__ inline const T& min(const T& a, const T& b, const T& c)
		{
			return min(min(a, b), c);
		}

		template <typename T>
		__device__ inline const T& max(const T& a, const T& b)
		{
			return a > b ? a : b;
		}

		template <typename T>
		__device__ inline const T& max(const T& a, const T& b, const T& c)
		{
			return max(max(a, b), c);
		}
	}
}

#endif /* VLR_RENDERING_UTIL */
