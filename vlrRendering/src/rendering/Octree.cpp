#include "rendering/Octree.h"

#include "maths/Colour.h"
#include "maths/Normal.h"
#include "rendering/child_desc.h"
#include "util/CUDAUtil.h"

#include <stdio.h>
#include <queue>
#include <vector>
#include <map>
#include <stdint.h>

#include <glm/glm.hpp>

namespace vlr
{
	namespace rendering
	{
		// Sort two child descriptor blocks by depth
		bool sortByDepth(child_desc_builder_block& lhs, child_desc_builder_block& rhs)
		{
			return lhs.depth < rhs.depth;
		}

		// Generate a node of tree
		void genNode(std::vector<child_desc_builder_block>&,
			point_test_func&,
			glm::vec3 min, glm::vec3 max, int depth,
			int max_depth, int block_id, int idx);

		int genOctree(int** ret, int max_depth,
			point_test_func& test_point_func,
			const glm::vec3& min, const glm::vec3& max)
		{
			const int child_desc_size = child_desc_size_ints * sizeof(int32_t);

			// Check for existence of root and immediately return if not
			glm::vec3 norm;
			glm::vec4 col;
			if (!test_point_func(min, max, norm, col))
			{
				*ret = nullptr;
				return 0;
			}

			// Generated blocks of child descriptors
			std::vector<child_desc_builder_block> child_desc_builder_blocks;

			// Generate tree
			// Create root child descriptor block
			child_desc_builder_block root;
			root.id = 0;
			root.count = 1;
			root.depth = 0;
			root.child_desc_builder[0].norm = norm;
			root.child_desc_builder[0].col = col;

			// Add to collection
			child_desc_builder_blocks.push_back(root);

			// Start generation at root
			genNode(child_desc_builder_blocks, test_point_func, min, max,
				0, max_depth, 0, 0);
			
			// Sort tree by depth
			std::sort(child_desc_builder_blocks.begin(), child_desc_builder_blocks.end(), sortByDepth);

			// A map of (id -> pointer) for child descriptor blocks
			std::map<int, pointer_desc> child_desc_builder_map;

			// A map of (freespace block -> first free space)
			std::map<int, int> first_freespace;

			// Iterate through and calculate pointers
			const int chunk_size = 0x8000;
			const int reserved_size = 0x4000;
			const int remaining_size = chunk_size - reserved_size;

			// Initial pointer += 4 to leave space for info block pointer
			size_t size = child_desc_size_ints;

			for (auto it = child_desc_builder_blocks.begin(); it != child_desc_builder_blocks.end(); ++it)
			{
				// Skip a slot for the info block pointer if this is at an 8kB boundary
				if ((size * child_desc_size) % 0x2000 == 0)
					size += 1;

				pointer_desc pointer;
				pointer.far = false;

				if ((size + it->count) % chunk_size > remaining_size)
				{
					size = chunk_size * (size / chunk_size + 1);
				}

				int ptr = size;

				// If this isn't the root
				if (it != child_desc_builder_blocks.begin())
				{
					// TODO: use parent pointer instead of parent block pointer
					int parent_ptr = child_desc_builder_map[it->parent_block_id].ptr;
					int relative_ptr = ptr - parent_ptr;

					if (relative_ptr > chunk_size)
					{
						int chunk_id = parent_ptr / chunk_size;
						int freespace_ptr = chunk_id * chunk_size + remaining_size;

						// Get first free space in parent chunk free space block
						if (first_freespace.count(chunk_id) == 0)
							first_freespace[chunk_id] = 0;

						int freespace_idx = first_freespace[chunk_id]++;
						freespace_ptr += freespace_idx;

						assert(freespace_idx < reserved_size);

						pointer.far = true;
						pointer.far_ptr = freespace_ptr;
					}
				}

				pointer.ptr = ptr;

				child_desc_builder_map[it->id] = pointer;

				size += it->count;
			}
			
			// Skip to start of next chunk
			size = chunk_size * (size / chunk_size + 1);

			// Allocate memory for tree
			int data_size = size * child_desc_size;
			int* data = (int*)malloc(data_size);

			int max_far = 0;

			// Iterate through blocks a second time and write data
			for (auto it = child_desc_builder_blocks.begin(); it != child_desc_builder_blocks.end(); ++it)
			{
				int cur_ptr_org = child_desc_builder_map[it->id].ptr;
				int cur_ptr = child_desc_builder_map[it->id].ptr;

				// For each child descriptor in block
				// Write it in reverse to match raycast
				for (int i = it->count-1; i >= 0; --i)
				{
					int* cur_descriptor = data + cur_ptr * child_desc_size_ints;

					// Initialise new child descriptor
					int desc = 0;
					
					// Get child pointer if this is a non-leaf
					int child_ptr = 0;
					
					bool far_needed = false;
					int far_ptr = 0;

					if ((it->child_desc_builder[i].non_leaf_mask & (1 << (7 - i))) != 0)
					{
						int child_id = it->child_desc_builder[i].child_id;

						child_ptr = child_desc_builder_map[child_id].ptr - cur_ptr;

						if (child_desc_builder_map[child_id].far)
						{
							// Indicate that this is a far pointer
							far_needed = true;

							// Write the far pointer
							int far_ptr_abs = child_desc_builder_map[child_id].far_ptr *
								child_desc_size_ints;
							far_ptr = child_desc_builder_map[child_id].far_ptr - cur_ptr;
							data[far_ptr_abs] = child_ptr;
						}
					}

					// Check child pointer is < 15 bits
					// (this should be ensured in the pointer mapping stage)
					if (!far_needed)
					{
						assert((unsigned int)child_ptr < chunk_size);
					
						// Write child pointer to descriptor
						// Child pointer is the relative pointer
						// between the parent (cur_ptr) and the child
						desc ^= child_ptr << 17;
					}
					else
					{
						assert((unsigned int)far_ptr < chunk_size);
					
						// Write child pointer to descriptor
						// Child pointer is the relative pointer
						// between the parent (cur_ptr) and the child
						desc ^= far_ptr << 17;

						int test = ((unsigned int)desc) >> 17;

						// Set far pointer flag
						desc ^= 1 << 16;
					}

					// Write normal to descriptor
					cur_descriptor[1] = (int)compressNormal(it->child_desc_builder[i].norm);

					// Write colour to descriptor
					cur_descriptor[2] = (int)compressColour(it->child_desc_builder[i].col);

					// Write child mask and non-leaf mask to descriptor
					desc ^= it->child_desc_builder[i].child_mask << 8;
					desc ^= it->child_desc_builder[i].non_leaf_mask;

					// Write data
					cur_descriptor[0] = desc;

					// Increment pointer
					cur_ptr += 1;
				}
			}

			const int colour = (int)0x00FF00FFu;

			//// Write info section pointers
			//for (uintptr_t i = 0; i < data_size; i += 0x2000)
			//{
			//	// If this isn't in space reserved for far pointers
			//	if (i % chunk_size < reserved_size)
			//	{
			//		// Write info section pointer every 0x2000 bytes
			//		int* info_ptr_loc = (int*)((int)data + i);

			//		printf("%d\n", *info_ptr_loc);

			//		*info_ptr_loc = colour;
			//	}
			//}

			*ret = data;

			return data_size;
		}

		void genNode(std::vector<child_desc_builder_block>& child_desc_builder_blocks,
			point_test_func& test_point_func,
			glm::vec3 min, glm::vec3 max, int depth, int max_depth,
			int block_id, int idx)
		{
			child_desc_builder_blocks[block_id].child_desc_builder[idx].child_mask = 0;
			child_desc_builder_blocks[block_id].child_desc_builder[idx].non_leaf_mask = 0;

			glm::vec3 half = 0.5f * (max - min);

			int nonleaf_count = 0;

			for (int i = 0; i < 8; ++i)
			{
				bool leaf = (depth + 1) >= max_depth;

				int x = (i & 1) >> 0;
				int y = (i & 2) >> 1;
				int z = (i & 4) >> 2;

				glm::vec3 new_min = min + glm::vec3(half.x * (float)x,
					half.y * (float)y, half.z * (float)z);
				glm::vec3 new_max = new_min + half;

				// Continue loop if this node is empty
				glm::vec3 norm;
				glm::vec4 col;
				if (!test_point_func(min, max, norm, col))
					continue;

				// Set colour
				child_desc_builder_blocks[block_id].child_desc_builder[idx].norm = norm;
				child_desc_builder_blocks[block_id].child_desc_builder[idx].col = col;

				// Set child bit
				child_desc_builder_blocks[block_id].child_desc_builder[idx].child_mask ^= 1 << (7 - i);

				// Continue with loop if this is a leaf
				if (leaf)
					continue;

				nonleaf_count++;

				// Set non-leaf bit
				child_desc_builder_blocks[block_id].child_desc_builder[idx].non_leaf_mask ^= 1 << (7 - i);
			}

			// If there are non-leaves, set child ptr
			// and generate children
			if (child_desc_builder_blocks[block_id].child_desc_builder[idx].non_leaf_mask != 0)
			{
				child_desc_builder_block child_block;
				child_block.id = child_desc_builder_blocks.size();
				child_block.count = nonleaf_count;
				child_block.depth = depth + 1;
				
				child_block.parent_id = idx;
				child_block.parent_block_id = block_id;

				child_desc_builder_blocks[block_id].child_desc_builder[idx].child_id = child_block.id;

				// Warning: this can invalidate desc reference
				// We re-set it since it is still needed
				child_desc_builder_blocks.push_back(child_block);
				//desc = child_desc_builder_blocks[block_id].child_desc_builder[idx];

				int nonleaves = 0;
				for (int i = 0; i < 8; ++i)
				{
					assert(nonleaves <= nonleaf_count);

					// If this is a leaf, continue
					if ((child_desc_builder_blocks[block_id].child_desc_builder[idx].non_leaf_mask & (1 << (7 - i))) == 0)
						continue;

					nonleaves++;

					int x = i & 1;
					int y = (i & 2) >> 1;
					int z = (i & 4) >> 2;

					glm::vec3 newMin = min + glm::vec3(half.x * (float)x,
						half.y * (float)y, half.z * (float)z);
					glm::vec3 newMax = newMin + half;

					genNode(child_desc_builder_blocks,
						test_point_func,
						newMin, newMax,
						depth + 1, max_depth,
						child_block.id, i);
				}
			}
		}
	}
}
