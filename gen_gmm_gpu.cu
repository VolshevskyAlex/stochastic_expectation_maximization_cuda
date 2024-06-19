#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include <vector>

#define BLOCK_SIZE1D	256
#define K				64

template <int N>
__global__ void generate_gmm_kernel(
	float *data,
	int data_pitch,
	int len,
	int distr_cnt,
	int distr_param_size,
	float *distr_params,
	curandStateMtgp32 *dv_rng_state)
{
	int tid = threadIdx.x;
	extern __shared__ float buf[];
	float *sm_rand_v = buf + distr_cnt*distr_param_size + tid;
	int ofs = BLOCK_SIZE1D * K * blockIdx.x + tid;

	for (int i = tid; i < distr_param_size*distr_cnt; i += BLOCK_SIZE1D)
		buf[i] = distr_params[i];

	__syncthreads();

	for (int i = 0; i < K && ofs < len; i++, ofs += BLOCK_SIZE1D)
	{
		int cluster_id = 0;
		{
			float v = curand_uniform(&dv_rng_state[blockIdx.x]);
			float *p_tau = buf;
			for (int j = 0; j < distr_cnt; j++, p_tau += distr_param_size)
			{
				if (v > *p_tau)
					cluster_id++;
				v -= *p_tau;
			}
		}
		float *p_rand_v = sm_rand_v;
#pragma unroll
		for (int j = 0; j < N; j++, p_rand_v += BLOCK_SIZE1D)
		{
			*p_rand_v = curand_normal(&dv_rng_state[blockIdx.x]);
		}

		float *p_dst = data + ofs;

		float *a = buf + cluster_id*distr_param_size + 1;
		// mu0, matrix_row0, mu1, matrix_row1, ...
#pragma unroll
		for (int j = 0; j < N; j++, p_dst += data_pitch)
		{
			float v = *a++; // mu
			p_rand_v = sm_rand_v;
#pragma unroll
			for (int k = 0; k <= j; k++, p_rand_v+= blockDim.x)
			{
				v += *a++ * *p_rand_v;
			}
			*p_dst = v;
		}
	}
}

template <int N>
void fill_data_gmm(float *data, int data_pitch, int len, std::vector<std::vector<float> > &gmm)
{
	cudaEvent_t start_, stop_;
	cudaEventCreate(&start_, 0);
	cudaEventCreate(&stop_, 0);

	int distr_cnt = (int)gmm.size();
	int cnt4block = BLOCK_SIZE1D * K;
	int block_cnt = (len + cnt4block - 1) / cnt4block;

	curandStateMtgp32 *d_state;
    cudaMalloc((void **)&d_state, block_cnt*sizeof(curandStateMtgp32));

	mtgp32_kernel_params *d_kernel_params;
    cudaMalloc((void **)&d_kernel_params, sizeof(mtgp32_kernel_params));

	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_kernel_params);
	curandMakeMTGP32KernelState(d_state, mtgp32dc_params_fast_11213, d_kernel_params, block_cnt, 1234);

	int distr_params_pack_size = 1 + N + N*(N+1)/2;
	int distr_param_size = distr_params_pack_size | 1;
	int sm_size = (distr_param_size*distr_cnt + BLOCK_SIZE1D*N) * sizeof(float);
	float *dv_params;
	cudaMalloc((void**)&dv_params, distr_param_size*distr_cnt*sizeof(float));
	for (int i = 0; i < distr_cnt; i++)
	{
		cudaMemcpy(dv_params + i*distr_param_size, gmm[i].data(), distr_params_pack_size*sizeof(float), cudaMemcpyHostToDevice);
	}

	generate_gmm_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(
		data,
		data_pitch,
		len, distr_cnt,
		distr_param_size,
		dv_params,
		d_state);

	cudaFree(d_state);
	cudaFree(d_kernel_params);
	cudaFree(dv_params);

	cudaEventDestroy(start_);
	cudaEventDestroy(stop_);
}

template
void fill_data_gmm<3>(float *data, int data_pitch, int len, std::vector<std::vector<float> > &gmm);
