#ifndef VLR_COMMON_UTIL
#define VLR_COMMON_UTIL

#include <vector>
#include <string>
#include <dirent.h>

namespace vlr
{
	namespace common
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
	}
}

#endif /* VLR_COMMON_UTIL */
