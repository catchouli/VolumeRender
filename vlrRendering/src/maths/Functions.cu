#include "maths/Functions.h"

#include <cuda.h>
#include <math_functions.h>

namespace vlr
{
	namespace rendering
	{
		__host__ __device__ int clamp(int x, int a, int b)
		{
			return x < a ? a : (x > b ? b : x);
		}
	}
}
