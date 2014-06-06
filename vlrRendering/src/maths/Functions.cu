#include "maths/Functions.h"

#include <cuda.h>
#include <math_functions.h>

namespace vlr
{
	namespace rendering
	{
		__host__ __device__ int32_t clamp(int32_t x, int32_t a, int32_t b)
		{
			return x < a ? a : (x > b ? b : x);
		}

		__host__ __device__ float sqr(float a)
		{
			return a * a;
		}
	}
}
