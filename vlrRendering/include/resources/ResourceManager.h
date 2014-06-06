#ifndef VLR_RESOURCEMANAGER_H
#define VLR_RESOURCEMANAGER_H

#include <map>
#include <string>

namespace vlr
{
	namespace rendering
	{
		class ResourceManager
		{
		public:
			template <typename T>
			static T* load(const char* name);

		protected:
			template <typename T>
			static T* loadResource(const char* name);
		};

		template<typename T>
		inline T* ResourceManager::load(const char* name)
		{
			std::string cppFilename(name);

			static std::map<std::string, T*> resources;

			// Load if not exists
			if (resources.find(cppFilename) == resources.end())
			{
				T* resource = loadResource<T>(name);

				if (resource == nullptr)
					return nullptr;

				resources[cppFilename] = resource;
			}

			return resources[cppFilename];
		}
	}
}

#endif /* VLR_RESOURCEMANAGER_H */
