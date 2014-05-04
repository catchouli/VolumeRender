#include <stdio.h>
#include <resources/Octree.h>
#include <resources/Mesh.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

using namespace vlr::rendering;

int main(int argc, char** argv)
{
	int depth = 5;

	char buffer[1024];

	const char* infile = "infile";
	const char* outfile = "";

	// Load command line arguments
	if (argc > 1)
		infile = argv[1];
	if (argc > 2)
		outfile = argv[2];
	if (argc > 3)
		depth = atoi(argv[3]);

	if (depth < 1)
	{
		fprintf(stderr, "Invalid depth\n");
		system("pause");

		return 1;
	}
	
	printf("In: %s\n", infile);
	printf("Out: %s\n", outfile);
	printf("Depth: %d\n", depth);

	// Load input mesh
	printf("Loading mesh...\n");

	Mesh mesh(true);
	if (!mesh.load(infile))
		return 1;

	// Rotate mesh (md2s are rotated weirdly for some reason)
	glm::mat4 rotation = glm::rotate(180.0f, glm::vec3(0, 0, 1.0f));
	rotation = glm::rotate(rotation, 90.0f, glm::vec3(0, 1.0f, 0));
	mesh.transform(rotation);

	// Generate tree
	printf("Generating tree...\n");

	int32_t* tree;
	int32_t len = genOctreeMesh(&tree, depth, &mesh);

	// Write tree to file
	FILE* file = fopen(outfile, "wb");

	if (file == nullptr)
	{
		fprintf(stderr, "Failed to open file for writing: %s\n", outfile);

		while (file == nullptr)
		{
			printf("Enter new filename: ", outfile);
			scanf("%1023s", buffer);
			printf("\nNew filename: %s\n", buffer);

			file = fopen(buffer, "wb");

			if (file == nullptr)
				fprintf(stderr, "Failed to open file for writing: %s\n", outfile);
		}
	}

	// Write to fd
	fwrite(tree, 1, len, file);

	printf("Finished writing tree\n");
	fclose(file);

	return 0;
}
