#ifndef VLR_RENDERING_MATRIX
#define VLR_RENDERING_MATRIX

#include <cuda_runtime_api.h>

namespace vlr
{
	namespace rendering
	{
		union float4
		{
			struct
			{
				float x, y, z, w;
			};

			float data[4];
		};

		union mat4
		{
			struct
			{
				float4 a, b, c, d;
			};

			float data[16];
		};

		__device__ __host__ inline void multMatrixVector(const mat4* mat,
			const float4* vec, float4* out)
		{
			const float* a = mat->data;
			const float* b = vec->data;
			float* res = out->data;

			res[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
			res[1] = a[4] * b[0] + a[5] * b[1] + a[6] * b[2] + a[7] * b[3];
			res[2] = a[8] * b[0] + a[9] * b[1] + a[10] * b[2] + a[11] * b[3];
			res[3] = a[12] * b[0] + a[13] * b[1] + a[14] * b[2] + a[15] * b[3];
		}
	}
}

#endif /* VLR_RENDERING_MATRIX */
