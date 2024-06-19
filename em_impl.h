/*
CUDA implementation of stochastic expectation maximization algorithm (SEM)
Author: A.Volshevsky entelecheia@yandex.ru
*/
#include <cuda_runtime.h>
#include <curand.h>

#include "em.h"
#include "k_means_impl.h"
#include "cov_inv.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

template <int N>
struct distr_param_expectation
{
	float k;
	float mu[N];
	float cov_inv[N*N];
};

template <int N>
class EM_gpu
{
public:
	enum {STAT_FLD_CNT = 1+N+N*(N+1)/2};	// 1-count, N-mean, N*(N-1)-covariance

private:
	int     data_length_;
	int     data_pitch_;
	float  *dv_data_;
	int    *dv_cluster_;
	float  *dv_randv_;
	distr_param_expectation<N>         *dv_distr_expec_;

	int                                 distr_cnt_;
	std::vector<distr_gauss_params<N> > gmm_params_;

	curandGenerator_t                   prngGPU_;
	reduce_buf                          reduce_b_;

	int    k_means_attempts_;
	int    k_means_max_iter_;
	float  k_means_eps_;

public:
	EM_gpu();
	~EM_gpu();

	void set_k_means_params(int attempts, int max_iter, float eps);
	void set_data(float *dv_data, int data_pitch, int len, int distr_cnt);
	bool estimate(
		std::vector<distr_gauss_params<N> > &gmm,
		k_means_visualizer &v1,
		em_visualizer<N> &v2);

private:
	void alloc_resources(int distr_cnt, int data_length);
	void free_resources();
	void calc_params();
	void update_expectation_param();
};

///
template <int N>
__global__ void maximization_step_kernel(
	const float *data,
	int data_pitch,
	const int *cluster_id,
	int cnt,
	int cluster_cnt,
	int sm_pitch,
	int row_cnt,
	float *stat,
	int stat_pitch)
{
	extern __shared__ float buf[];

	int tid = threadIdx.x;
	int buf_sz = row_cnt << 4;
	
	for (int i = tid; i < buf_sz; i+= BLOCK_SIZE1D)
	{
		buf[i] = 0.0f;
	}
	__syncthreads();

	float *p_data = buf + buf_sz + tid;
	int ofs = BLOCK_SIZE1D * BLOCK_SIZE1D * blockIdx.x + tid;
	for (int i = 0; i < BLOCK_SIZE1D && ofs < cnt; i++, ofs+= BLOCK_SIZE1D)
	{
		const float *p_gm  = data + ofs;
		float *p_acc = buf + (tid & 15) + (cluster_id[ofs] << 4);
		atomicAdd(p_acc, 1.0f);
#pragma unroll
		for (int j = 0; j < N*BLOCK_SIZE1D; j+= BLOCK_SIZE1D, p_gm += data_pitch)
		{
			float t = *p_gm;
			p_data[j] = t;

			p_acc += sm_pitch;
			atomicAdd(p_acc, t);
		}

#pragma unroll
		for (int j = 0; j < N*BLOCK_SIZE1D; j+= BLOCK_SIZE1D)
		{
			float t = p_data[j];
#pragma unroll
			for (int k = j; k < N*BLOCK_SIZE1D; k+= BLOCK_SIZE1D)
			{
				p_acc += sm_pitch;
				atomicAdd(p_acc, t*p_data[k]);
			}
		}
	}
	__syncthreads();

	{
		float *p = buf + tid;
		for (int j = 0; j < row_cnt; j++, p+= 16)
		{
#pragma unroll
			for (int step = 8; step > 0; step >>= 1)
			{
				if (tid < step)
					p[0] += p[step];
				__syncthreads();
			}
		}
	}
	float *out_p = stat + blockIdx.x;
	for (int j = tid; j < row_cnt; j+= BLOCK_SIZE1D)
	{
		out_p[j*stat_pitch] = buf[j<<4];
	}
}

template <int N>
__forceinline__ __device__ float get_prob(const distr_param_expectation<N> *d, float *v)
{
	const float *a  = d->cov_inv;

#pragma unroll
	for (int i = 0; i < N; i++)
	{
		v[i*BLOCK_SIZE1D] -= d->mu[i];
	}

	float s = 0.f;
#pragma unroll
	for (int i = 0; i < N*BLOCK_SIZE1D; i+= BLOCK_SIZE1D)
	{
		float t = 0.f;
#pragma unroll
		for (int j = 0; j < N*BLOCK_SIZE1D; j+= BLOCK_SIZE1D)
		{
			t+= *a++ * v[j];
		}
		s+= t * v[i];
	}

	return expf(-0.5f * s + d->k);
}

template <int N>
__global__ void expectation_step_likelihood_kernel(
	const float *data,
	int data_pitch,
	const distr_param_expectation<N> *gm_distr,
	int distr_cnt,
	const float *rand_v,
	int *cluster_id,
	float *likelihood_in_block,
	int cnt)
{
	typedef distr_param_expectation<N> distr_param;

	int tid = threadIdx.x;
	int ofs = blockIdx.x*blockDim.x + tid;

	const int distr_ar_size = distr_cnt*(sizeof(distr_param)/sizeof(float));

	extern __shared__ float buf[];
	for (int i = tid; i < distr_ar_size; i+= BLOCK_SIZE1D)
	{
		buf[i] = ((float*)gm_distr)[i];
	}
	__syncthreads();

	float *likelihood_a = buf + distr_ar_size;
	float *p_sm_data0   = likelihood_a + BLOCK_SIZE1D + tid;
	float *p_sm_prob0   = p_sm_data0 + N * BLOCK_SIZE1D;

	if (ofs < cnt)
	{
		data += ofs;
#pragma unroll
		for (int i = 0; i < N*BLOCK_SIZE1D; i+= BLOCK_SIZE1D)
		{
			p_sm_data0[i] = *data;
			data += data_pitch;
		}

		float prob_sum = 0.0f;
		float *p_sm_prob = p_sm_prob0;
		distr_param *p_D = reinterpret_cast<distr_param*>(buf);
		for (int i = 0; i < distr_cnt; i++, p_D++, p_sm_prob += BLOCK_SIZE1D)
		{
			prob_sum += get_prob(p_D, p_sm_data0);
			*p_sm_prob = prob_sum;
		}
		likelihood_a[tid] = log(prob_sum);

		prob_sum *= rand_v[ofs];
		int c_id = 0;

		p_sm_prob = p_sm_prob0;
		for (int i = 0; i < distr_cnt; i++, p_sm_prob += BLOCK_SIZE1D)
		{
			if (*p_sm_prob < prob_sum)
				c_id++;
		}
		cluster_id[ofs] = c_id;
	}
	else
	{
		likelihood_a[tid] = 0.0f;
	}
	__syncthreads();

	float *p_sm = likelihood_a + tid;
	for (int step = BLOCK_SIZE1D/2; step > 0; step >>= 1)
	{
		if (tid < step)
		{
			p_sm[0] += p_sm[step];
		}
		__syncthreads();
	}
	if (tid==0)
	{
		likelihood_in_block[blockIdx.x] = likelihood_a[0];
	}
}

///
template <int N>
void lanch_maximization_step(
	const float *data,
	int data_pitch,
	const int *cluster_id,
	int data_length,
	int cluster_cnt,
	reduce_buf &statistic_b)
{
	int stat_fld_cnt = EM_gpu<N>::STAT_FLD_CNT;
	int block_cnt = (data_length + BLOCK_SIZE1D*BLOCK_SIZE1D - 1) / (BLOCK_SIZE1D*BLOCK_SIZE1D);

	int row_cnt = cluster_cnt*stat_fld_cnt;
	int sm_pitch = 16*cluster_cnt;
	int sm_size  = (sm_pitch*stat_fld_cnt + N*BLOCK_SIZE1D) * sizeof(float);

	maximization_step_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(
		data,
		data_pitch,
		cluster_id,
		data_length,
		cluster_cnt,
		sm_pitch,
		row_cnt,
		statistic_b.buf0,
		statistic_b.pitch4);

	reduce_internal(statistic_b, block_cnt, row_cnt);
}

template <int N>
void lanch_expectation_step_likelihood(
	const float *data,
	int data_pitch,
	const distr_param_expectation<N> *distr,
	int distr_cnt,
	const float *randv,
	int *cluster,
	int data_length,
	reduce_buf &likelihood_b)
{
	typedef distr_param_expectation<N> distr_param;

	int block_cnt = (data_length + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;
	int sm_size = distr_cnt*sizeof(distr_param) + (BLOCK_SIZE1D + N*BLOCK_SIZE1D + distr_cnt*BLOCK_SIZE1D)*sizeof(float);

#if 0
//	test on CPU
	std::vector<distr_param_expectation<N> > a_distr(distr_cnt);
	cudaMemcpy(&a_distr[0], distr, sizeof(distr_param_expectation<N>)*distr_cnt, cudaMemcpyDeviceToHost);
	std::vector<float> a_data(data_pitch*N);
	cudaMemcpy(&a_data[0], data, sizeof(float)*data_pitch*N, cudaMemcpyDeviceToHost);

	float liklehood = 0.f;
	for (int ii = 0; ii < data_length; ii++)
	{
		float v[N];
		for (int i = 0; i < N; i++)
		{
			v[i] = a_data[ii + i*data_pitch];
		}

		float p_sum = 0.f;
		for (int d_ix = 0; d_ix < distr_cnt; d_ix++)
		{
			distr_param *d = &a_distr[d_ix];
			const float *a  = d->cov_inv;

			for (int i = 0; i < N; i++)
			{
				v[i] -= d->mu[i];
			}

			float s = 0.f;
			for (int i = 0; i < N; i++)
			{
				float t = 0.f;
				for (int j = 0; j < N; j++)
				{
					t+= *a++ * v[j];
				}
				s+= t * v[i];
			}

			float prob = expf(-0.5f * s + d->k);
			p_sum += prob;
		}
		liklehood += logf(p_sum);
	}
	liklehood /= data_length;
	float lll = expf(liklehood);
#endif

	expectation_step_likelihood_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(
		data,
		data_pitch,
		distr,
		distr_cnt,
		randv,
		cluster,
		likelihood_b.buf0,
		data_length);

	reduce_internal(likelihood_b, block_cnt, 1);
}

///
template <int N>
EM_gpu<N>::EM_gpu()
{
	dv_data_    = NULL;
	dv_cluster_ = NULL;
	dv_randv_   = NULL;
	dv_distr_expec_ = NULL;
	prngGPU_        = NULL;
	memset(&reduce_b_, 0, sizeof(reduce_buf));

	set_k_means_params(5, 400, 1e-3f);
}

template <int N>
EM_gpu<N>::~EM_gpu()
{
	free_resources();
}

template <int N>
void EM_gpu<N>::set_k_means_params(int attempts, int max_iter, float eps)
{
	k_means_attempts_ = attempts;
	k_means_max_iter_ = max_iter;
	k_means_eps_      = eps;
}

template <int N>
void EM_gpu<N>::set_data(float *dv_data, int data_pitch, int data_length, int distr_cnt)
{
	free_resources();
	alloc_resources(distr_cnt, data_length);
	dv_data_    = dv_data;
	data_pitch_ = data_pitch;
}

template <int N>
bool EM_gpu<N>::estimate(
	std::vector<distr_gauss_params<N> > &gmm,
	k_means_visualizer                  &k_means_v,
	em_visualizer<N>                    &em_v)
{
	std::vector<float> means_centers;
	if (!k_means<N>(
		dv_data_,
		data_pitch_,
		dv_cluster_,
		data_length_,
		distr_cnt_,
		means_centers,
		k_means_attempts_,
		k_means_max_iter_,
		k_means_eps_,
		k_means_v))
	{
		return false;
	}

	gmm_params_.resize(distr_cnt_);

	float prev_likelihood = -FLT_MAX;

#ifdef TIME_MEASURE
	cudaEvent_t start_, stop_;
	cudaEventCreate(&start_, 0);
	cudaEventCreate(&stop_, 0);

	float tm_maximization_step = 0.f;
	float tm_GenerateUniform   = 0.f;
	float tm_expectation_step  = 0.f;
	float tm0, tm1, tm2;
#endif

	bool interrupted = false;
	int iter = 0, max_iter;
	do {
		TIME_START();
		lanch_maximization_step<N>(dv_data_, data_pitch_, dv_cluster_, data_length_, distr_cnt_, reduce_b_);
		TIME_STOP(tm0);

		calc_params();
		update_expectation_param();

		TIME_START();
		curandGenerateUniform(prngGPU_, dv_randv_, data_length_);
		TIME_STOP(tm1);

		TIME_START();
		lanch_expectation_step_likelihood<N>(dv_data_, data_pitch_, dv_distr_expec_, distr_cnt_, dv_randv_, dv_cluster_, data_length_, reduce_b_);
		TIME_STOP(tm2);

		float likelihood;
		cudaMemcpy(&likelihood, reduce_b_.buf0, sizeof(float), cudaMemcpyDeviceToHost);
		likelihood /= data_length_;

#ifdef TIME_MEASURE
		printf(
		"\ttm_maximization_step = %f\n"
		"\ttm_GenerateUniform = %f\n"
		"\ttm_expectation_step = %f\n", tm0, tm1, tm2);
		tm_maximization_step += tm0;
		tm_GenerateUniform   += tm1;
		tm_expectation_step  += tm2;
#endif

		em_v.visualize(gmm_params_, ++iter, likelihood, interrupted);

		if (likelihood > prev_likelihood)
		{
			gmm = gmm_params_;
			prev_likelihood = likelihood;
			max_iter = iter + 5;
		}
	}
	while (iter < max_iter && !interrupted);

#ifdef TIME_MEASURE
	printf(
	"tm_maximization_step = %f\n"
	"tm_GenerateUniform = %f\n"
	"tm_expectation_step = %f\n", tm_maximization_step, tm_GenerateUniform, tm_expectation_step);
	cudaEventDestroy(start_);
	cudaEventDestroy(stop_);
#endif

	return !interrupted;
}

template <int N>
void EM_gpu<N>::calc_params()
{
	int row_cnt = STAT_FLD_CNT * distr_cnt_;
	std::vector<float> a(row_cnt);
	cudaMemcpy2D(&a[0], sizeof(float), reduce_b_.buf0, reduce_b_.pitch4*sizeof(float), sizeof(float), row_cnt, cudaMemcpyDeviceToHost);

	for (int ix = 0; ix < distr_cnt_; ix++)
	{
		float *p = a.data() + ix;
		distr_gauss_params<N> &dst_it = gmm_params_[ix];

		double sampl_cnt = p[0];
		p += distr_cnt_;
		assert(sampl_cnt!=0);
		double w = 1.0 / sampl_cnt;
		for (int i = 0; i < N; i++)
		{
			dst_it.mu[i] = w * p[0];
			p += distr_cnt_;
		}
		for (int i = 0; i < N; i++)
		{
			for (int j = i; j < N; j++)
			{
				dst_it.cov[i*N+j] = w * p[0];
				p += distr_cnt_;
			}
		}
		for (int i = 1; i < N; i++)
		{
			for (int j = 0; j < i; j++)
			{
				dst_it.cov[i*N+j] = dst_it.cov[j*N+i];
			}
		}
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				dst_it.cov[i*N+j] -= dst_it.mu[i]*dst_it.mu[j];
			}
		}
		calc_inverse_cov(dst_it.cov, dst_it.cov_inv, dst_it.cov_determinant, N);
		dst_it.tau = sampl_cnt / data_length_;
	}
}

template <int N>
void EM_gpu<N>::update_expectation_param()
{
	std::vector<distr_param_expectation<N> > a(distr_cnt_);
	for (int i = 0; i < distr_cnt_; i++)
	{
		const distr_gauss_params<N> &it_src = gmm_params_[i];
		distr_param_expectation<N> &it = a[i];
		it.k  = static_cast<float>(log(it_src.tau / (pow(2.0*M_PI, N/2.0) * sqrt(it_src.cov_determinant))));
		for (int j = 0; j < N; j++)
		{
			it.mu[j] = static_cast<float>(it_src.mu[j]);
			for (int k = 0; k < N; k++)
			{
				it.cov_inv[N*j+k] = static_cast<float>(it_src.cov_inv[N*j+k]);
			}
		}
	}
	for (int i = distr_cnt_-1; i > 0; i--)
	{
		for (int j = 0; j < N; j++)
		{
			a[i].mu[j] -= a[i-1].mu[j];
		}
	}
	cudaMemcpy(dv_distr_expec_, &a[0], sizeof(distr_param_expectation<N>)*distr_cnt_, cudaMemcpyHostToDevice);
}

template <int N>
void EM_gpu<N>::alloc_resources(int distr_cnt, int data_length)
{
	distr_cnt_ = distr_cnt;
	data_length_ = data_length;

	cudaMalloc((void**)&dv_cluster_, data_length_*sizeof(int));
	cudaMalloc((void**)&dv_randv_  , data_length_*sizeof(float));
	cudaMalloc((void**)&dv_distr_expec_, distr_cnt*sizeof(distr_param_expectation<N>));

	int block_cnt = (data_length_ + BLOCK_SIZE1D - 1) / (BLOCK_SIZE1D);

	alloc_reduce_buf(&reduce_b_, block_cnt, distr_cnt_*EM_gpu<N>::STAT_FLD_CNT);

	curandCreateGenerator(&prngGPU_, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(prngGPU_, 1234);
}

template <int N>
void EM_gpu<N>::free_resources()
{
	cudaFree(dv_cluster_);
	cudaFree(dv_distr_expec_);
	free_reduce_buf(&reduce_b_);
	curandDestroyGenerator(prngGPU_);
}

template <int N>
void expectation_maximization(
	float *dv_data,
	int data_pitch,
	int data_length,
	int distr_cnt,
	std::vector<distr_gauss_params<N> > &d,
	k_means_visualizer &v1,
	em_visualizer<N>   &v2)
{
	EM_gpu<N> em;
	em.set_data(dv_data, data_pitch, data_length, distr_cnt);
	em.estimate(d, v1, v2);
}
