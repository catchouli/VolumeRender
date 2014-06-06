#include "application/VolumeRender.h"

#include "util/Util.h"
#include "util/CUDAUtil.h"

#include "resources/Octree.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

using namespace vlr::rendering;

namespace vlr
{
	void VolumeRender::generate()
	{
		double start_time, end_time, dt;

		start_time = glfwGetTime();

		// Load tree
		int32_t tree_size = 0;
		tree_data = 0;
		
		// Load from file
		if (_scene == 0)
		{
			tree_size = rendering::read_full_file_binary(_treeFilename, &tree_data);

			if (tree_size == 0)
			{
				fprintf(stderr, "Invalid tree file: %s\n", _treeFilename);
				exit(1);
			}
		}
		else
		{
			char filename_buffer[1024];
			sprintf(filename_buffer, "scene%d.tree.%d", _scene, _depth);

			FILE* file = fopen(filename_buffer, "rb");

			if (file != nullptr)
			{
				// Load tree
				printf("Loading file %s\n", filename_buffer);
				fclose(file);

				tree_size = rendering::read_full_file_binary(filename_buffer, &tree_data);

				if (tree_size == 0)
				{
					fprintf(stderr, "Invalid tree file: %s\n", filename_buffer);
					exit(1);
				}
			}
			else
			{
				// Generate
				printf("Generating scene for the first time, please wait..\n");

				// Generate from sphere
				if (_scene == 1)
				{
					tree_size =
						rendering::genOctreeSphere((int32_t**)&tree_data, _depth,
						glm::vec3(0.5f, 0.5f, 0.5f), 0.4f);
				}

				// Generate from miku mesh
				if (_scene == 2)
				{
					if (_mesh.load("miku.md2"))
					{
						// Rotate mesh to right rotation (md2s are messed up like that..)
						glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
						rotation = glm::rotate(rotation, 90.0f, glm::vec3(0, 1.0f, 0));
						_mesh.transform(rotation);

						tree_size = genOctreeMesh((int32_t**)&tree_data, _depth, &_mesh);
					}
				}

				// r = 1.5 sphere with embedded r = 1 sphere
				if (_scene == 3)
				{
					if (_mesh.load("miku.md2"))
					{
						// Rotate mesh to right rotation (md2s are messed up like that..)
						glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
						rotation = glm::rotate(rotation, 90.0f, glm::vec3(0, 1.0f, 0));
						_mesh.transform(rotation);
			
						// Get min, max and func for octree generation
						glm::vec3 overall_min = *_mesh.getMin();
						glm::vec3 overall_max = *_mesh.getMax();
						overall_max += (overall_max - overall_min);

						// Make cube around bounds
						glm::vec3 extents = (overall_max - overall_min) * 0.5f;
						glm::vec3 centre = overall_min + extents;

						float greatest_extent = std::max(std::max(extents.x, extents.y), extents.z);
			
						extents.x = greatest_extent;
						extents.y = greatest_extent;
						extents.z = greatest_extent;
			
						overall_min = centre - extents;
						overall_max = centre + extents;

						// Sphere position
						glm::vec3 sphere_pos = centre;
						sphere_pos.x += extents.x * 0.5f;
						float sphere_radius = 10.0f;

						// Test function
						auto test_func = [&] (glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
						{
							glm::vec3 half_size = 0.5f * (max - min);
							glm::vec3 centre = min + half_size;
							float half_size_one_axis = half_size.x;

							//bool sphere_intersect = cubeSphereSurfaceIntersection(centre, half_size_one_axis, sphere_pos, sphere_radius);
							bool sphere_intersect = boxSphereIntersection(min, max, sphere_pos, sphere_radius);

							if (sphere_intersect)
							{
								// This is not normalised to save generation time
								// it is normalised later in the GPU anyway after being unpacked
								shading_attributes.normal = sphere_pos - centre;
								shading_attributes.colour = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);
								shading_attributes.reflectivity = 1.0f;

								shading_attributes.refractive_index = 1.5f;

								float distFromCentre = glm::length(shading_attributes.normal);

								if (distFromCentre < 0.5f * sphere_radius)
									shading_attributes.refractive_index = 1.0f;

								return true;
							}
							else
							{
								bool mesh_intersect = meshAABBIntersect(&_mesh, min, max, shading_attributes);

								if (mesh_intersect)
								{
									shading_attributes.colour.a = 1.0f;
									shading_attributes.reflectivity = 0.0f;
									return true;
								}
							}

							return false;
						};

						point_test_func func(test_func);

						// Generate octree
						tree_size = genOctree((int32_t**)&tree_data, _depth, func, overall_min, overall_max);
					}
				}

				// r = 1.5 sphere with hollow sphere (should be the same as above)
				if (_scene == 4)
				{
					if (_mesh.load("miku.md2"))
					{
						// Rotate mesh to right rotation (md2s are messed up like that..)
						glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
						rotation = glm::rotate(rotation, 90.0f, glm::vec3(0, 1.0f, 0));
						_mesh.transform(rotation);
			
						// Get min, max and func for octree generation
						glm::vec3 overall_min = *_mesh.getMin();
						glm::vec3 overall_max = *_mesh.getMax();
						overall_max += (overall_max - overall_min);

						// Make cube around bounds
						glm::vec3 extents = (overall_max - overall_min) * 0.5f;
						glm::vec3 centre = overall_min + extents;

						float greatest_extent = std::max(std::max(extents.x, extents.y), extents.z);
			
						extents.x = greatest_extent;
						extents.y = greatest_extent;
						extents.z = greatest_extent;
			
						overall_min = centre - extents;
						overall_max = centre + extents;

						// Sphere position
						glm::vec3 sphere_pos = centre;
						sphere_pos.x += extents.x * 0.5f;
						float sphere_radius = 10.0f;

						// Test function
						auto test_func = [&] (glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
						{
							glm::vec3 half_size = 0.5f * (max - min);
							glm::vec3 centre = min + half_size;
							float half_size_one_axis = half_size.x;

							//bool sphere_intersect = cubeSphereSurfaceIntersection(centre, half_size_one_axis, sphere_pos, sphere_radius);
							bool sphere_intersect = boxSphereIntersection(min, max, sphere_pos, sphere_radius);

							if (sphere_intersect)
							{
								// This is not normalised to save generation time
								// it is normalised later in the GPU anyway after being unpacked
								shading_attributes.normal = sphere_pos - centre;
								shading_attributes.colour = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);
								shading_attributes.reflectivity = 1.0f;

								shading_attributes.refractive_index = 1.5f;

								float distFromCentre = glm::length(shading_attributes.normal);

								if (distFromCentre < 0.5f * sphere_radius)
									return false;


								return true;
							}
							else
							{
								bool mesh_intersect = meshAABBIntersect(&_mesh, min, max, shading_attributes);

								if (mesh_intersect)
								{
									shading_attributes.colour.a = 1.0f;
									shading_attributes.reflectivity = 0.0f;
									return true;
								}
							}

							return false;
						};

						point_test_func func(test_func);

						// Generate octree
						tree_size = genOctree((int32_t**)&tree_data, _depth, func, overall_min, overall_max);
					}
				}

				// Generate from mesh
				if (_scene == 5)
				{
					Mesh teapot("teapot.obj", true);
					Mesh checkerboard("checkerboard.obj", true);
					
					// Lift teapot off checkerboard a bit so they don't intersect
					glm::mat4 translation = glm::translate(glm::vec3(0.0f, 0.1f, 0.0f));
					glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
					checkerboard.transform(translation * rotation);

					teapot.transform(rotation);
			
					// Get min, max and func for octree generation
					glm::vec3 overall_min = *teapot.getMin();
					glm::vec3 overall_max = *teapot.getMax();
					overall_max += (overall_max - overall_min);

					// Make cube around bounds
					glm::vec3 extents = (overall_max - overall_min) * 0.5f;
					glm::vec3 centre = overall_min + extents;

					float greatest_extent = std::max(std::max(extents.x, extents.y), extents.z);
			
					extents.x = greatest_extent;
					extents.y = greatest_extent;
					extents.z = greatest_extent;
			
					overall_min = centre - extents;
					overall_max = centre + extents;

					// Test function
					auto test_func = [&] (glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
					{
						if (meshAABBIntersect(&checkerboard, min, max, shading_attributes))
						{
							shading_attributes.colour.a = 1.0f;
							shading_attributes.reflectivity = 0.0f;

							return true;
						}

						if (meshAABBIntersect(&teapot, min, max, shading_attributes))
						{
							shading_attributes.colour.a = 0.3f;
							shading_attributes.reflectivity = 0.5f;

							return true;
						}

						return false;
					};

					point_test_func func(test_func);

					// Generate octree
					tree_size = genOctree((int32_t**)&tree_data, _depth, func, overall_min, overall_max);
				}

				// transparent and reflective boxes on checkerboard
				if (_scene == 6)
				{
					Mesh checkerboard("checkerboard.obj", true);

					// The cube should have flat normals instead of smooth
					Mesh box1("cube.obj", true, aiProcess_GenNormals);
					Mesh box2("cube.obj", true, aiProcess_GenNormals);

					// Rotate checkerboard
					glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
					checkerboard.transform(rotation);

					// Scale and position box
					glm::mat4 translation = glm::translate(glm::vec3(0, 1.0f, 0));
					glm::mat4 scale = glm::scale(glm::vec3(0.25f));
					glm::mat4 box_transform = scale * translation;
					box1.transform(scale);
					box2.transform(scale);
			
					// Get min, max and func for octree generation
					glm::vec3 overall_min = *checkerboard.getMin();
					glm::vec3 overall_max = *checkerboard.getMax();
					overall_max += (overall_max - overall_min);

					// Make cube around bounds
					glm::vec3 extents = (overall_max - overall_min) * 0.5f;
					glm::vec3 centre = overall_min + extents;

					float greatest_extent = std::max(std::max(extents.x, extents.y), extents.z);
			
					extents.x = greatest_extent;
					extents.y = greatest_extent;
					extents.z = greatest_extent;
			
					overall_min = centre - extents;
					overall_max = centre + extents;

					// Test function
					auto test_func = [&] (glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
					{
						if (meshAABBIntersect(&checkerboard, min, max, shading_attributes))
						{
							shading_attributes.colour.a = 1.0f;
							shading_attributes.reflectivity = 0.0f;

							return true;
						}

						if (meshAABBIntersect(&box1, min, max, shading_attributes))
						{
							shading_attributes.colour.a = 1.0f;
							shading_attributes.reflectivity = 1.0f;

							return true;
						}

						return false;
					};

					point_test_func func(test_func);

					// Generate octree
					tree_size = genOctree((int32_t**)&tree_data, _depth, func, overall_min, overall_max);
				}

				// Reflective inverted box with teapot inside
				if (_scene == 7)
				{
					// The cube should have flat normals instead of smooth
					Mesh teapot("teapot.obj", true);
					Mesh box1("inverted_cube.obj", true, aiProcess_GenNormals);

					teapot.transform(glm::scale(glm::vec3(0.2f)) * glm::rotate(180.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
			
					// Get min, max and func for octree generation
					glm::vec3 overall_min = *box1.getMin();
					glm::vec3 overall_max = *box1.getMax();
					overall_max += (overall_max - overall_min);

					// Make cube around bounds
					glm::vec3 extents = (overall_max - overall_min) * 0.5f;
					glm::vec3 centre = overall_min + extents;

					float greatest_extent = std::max(std::max(extents.x, extents.y), extents.z);
			
					extents.x = greatest_extent;
					extents.y = greatest_extent;
					extents.z = greatest_extent;
			
					overall_min = centre - extents;
					overall_max = centre + extents;

					// Test function
					auto test_func = [&] (glm::vec3 min, glm::vec3 max, raw_attachment_uncompressed& shading_attributes)
					{
						glm::vec3 aabb_extents = (max - min) * 0.5f;
						glm::vec3 aabb_centre = aabb_centre + min;

						if (meshAABBIntersect(&box1, min, max, shading_attributes))
						{
							shading_attributes.colour.a = 1.0f;
							shading_attributes.reflectivity = 1.0f;

							return true;
						}

						if (meshAABBIntersect(&teapot, min, max, shading_attributes))
						{
							shading_attributes.colour = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
							shading_attributes.reflectivity = 0.0f;

							return true;
						}

						return false;
					};

					point_test_func func(test_func);

					// Generate octree
					tree_size = genOctree((int32_t**)&tree_data, _depth, func, overall_min, overall_max);
				}

				// Write custom generation to file
				if (_saveTrees)
				{
					printf("Writing %s...\n", filename_buffer);

					file = fopen(filename_buffer, "wb");

					if (file == nullptr)
					{
						fprintf(stderr, "Failed to open file for writing: %s\n", filename_buffer);
					}
					else
					{
						fwrite(tree_data, tree_size, sizeof(char), file);
						fclose(file);
					}
				}
			}
		}
		
		// Calculate time to load/generate
		end_time = glfwGetTime();
		dt = end_time - start_time;

		printf("Time to load/generate tree: %f\n", dt);
		printf("%.2fMB\n", tree_size / (1024.0f * 1024.0f));

		// Reset timer
		start_time = glfwGetTime();

		// Upload sphere to GPU
		gpuErrchk(cudaMalloc((void**)&_gpuTree, tree_size));
		gpuErrchk(cudaMemcpy(_gpuTree, tree_data, tree_size, cudaMemcpyHostToDevice));

		// Calculate time to upload
		end_time = glfwGetTime();
		dt = end_time - start_time;

		printf("Time to upload tree: %f\n", dt);
		printf("%.2fMB\n", tree_size / (1024.0f * 1024.0f));

		// Free CPU memory
		//free(tree_data);
	}
}