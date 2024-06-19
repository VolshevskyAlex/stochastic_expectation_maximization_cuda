#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE1D 512

__global__ void cvt_kernel(float4 *dst, float *src1, int src1_pitch, float *src2, int data_len)
{
	int ofs = blockDim.x * blockIdx.x + threadIdx.x;
	if (ofs < data_len)
	{
		dst[ofs] = make_float4(src1[ofs], src1[ofs+src1_pitch], src1[ofs+src1_pitch+src1_pitch], src2[ofs]);
	}
}

void lanch_cvt(float4 *dst, float *src1, int src1_pitch, float *src2, int data_len)
{
	int block_cnt = (data_len + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;
	cvt_kernel<<<block_cnt, BLOCK_SIZE1D>>>(dst, src1, src1_pitch, src2, data_len);
}
