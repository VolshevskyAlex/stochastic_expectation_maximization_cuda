/*
CUDA implementation of k-means algorithm
Author: A.Volshevsky entelecheia@yandex.ru
*/
#include <cuda_runtime.h>
#include <thrust/reduce.h>

#include "k_means.h"

#define BLOCK_SIZE1D 512

struct reduce_buf
{
	float *buf0;
	float *buf1;
	int    capacity;
	int    row_cnt;
	size_t pitch4;	// pitch in float
};

///
template <int N>
static __global__ void assign_nearest_cluster_kernel(
	const float *data,
	int          data_pitch,
	const float *cluster_centers,
	int distr_cnt,
	int *cluster_id,
	int cnt)
{
	extern __shared__ float buf[];
	int centers_buf_size = N*distr_cnt;
	float *sm_centers = buf;
	float *sm_data = buf + centers_buf_size;
	int tid = threadIdx.x;
	int ofs = BLOCK_SIZE1D*blockIdx.x + tid;

	for (int i = tid; i < centers_buf_size; i += BLOCK_SIZE1D)
	{
		sm_centers[i] = cluster_centers[i];
	}
	__syncthreads();

	if (ofs < cnt)
	{
		const float *p = data + ofs;
		float *p_sm_data = sm_data + tid;
#pragma unroll
		for (int j = 0; j < N; j++, p_sm_data += BLOCK_SIZE1D, p+= data_pitch)
		{
			*p_sm_data = *p;
		}

		int cl_id = 0;
		float min_dist = FLT_MAX;
		for (int i = 0; i < distr_cnt; i++)
		{
			p_sm_data = sm_data + tid;
			float dist = 0.f;
#pragma unroll
			for (int j = 0; j < N; j++, p_sm_data += BLOCK_SIZE1D, sm_centers++)
			{
				float delta = *sm_centers - *p_sm_data;
				dist += delta*delta;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				cl_id = i;
			}
		}
		cluster_id[ofs] = cl_id;
	}	
}

template <int N>
static __global__ void cluster_centers_kernel(
	const float *data,
	int          data_pitch,
	const int   *cluster_id,
	int cnt,
	int cluster_cnt,
	int sm_pitch,
	int row_cnt,
	float *stat,
	int stat_pitch)
{
	extern __shared__ float buf[];
	
	int tid = threadIdx.x;
	
	for (int i = tid, buf_sz = row_cnt << 5; i < buf_sz; i+= BLOCK_SIZE1D)
	{
		buf[i] = 0.0f;
	}
	__syncthreads();

	int ofs = BLOCK_SIZE1D * BLOCK_SIZE1D * blockIdx.x + tid;
	for (int i = 0; i < BLOCK_SIZE1D && ofs < cnt; i++, ofs+= BLOCK_SIZE1D)
	{
		const float *p_data = data + ofs;

		float *p = buf + (tid & 31) + (cluster_id[ofs] << 5);
		atomicAdd(p, 1.0f);
#pragma unroll
		for (int j = 0; j < N; j++, p_data += data_pitch)
		{
			p += sm_pitch;
			atomicAdd(p, *p_data);
		}
	}
	__syncthreads();

	{
		float *p = buf + tid;
		for (int j = 0; j < row_cnt; j++, p+= 32)
		{
#pragma unroll
			for (int step = 16; step > 0; step >>= 1)
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
		out_p[j*stat_pitch] = buf[j<<5];
	}
}

static __global__ void reduce_kernel(
	float *src_data,
	float *dst_data,
	int cnt)
{
	__shared__ float a[BLOCK_SIZE1D];

	int tid = threadIdx.x;
	int ofs = (2*BLOCK_SIZE1D) * blockIdx.x + tid;

	if (ofs+BLOCK_SIZE1D < cnt)
	{
		a[tid] = src_data[ofs] + src_data[ofs+BLOCK_SIZE1D];
	}
	else if (ofs < cnt)
	{
		a[tid] = src_data[ofs];
	}
	else
	{
		a[tid] = 0.0f;
	}
	__syncthreads();

	for (int step = BLOCK_SIZE1D/2; step > 0; step >>= 1)
	{
		if (tid < step)
		{
			float *p_sm = a + tid;
			p_sm[0] += p_sm[step];
		}
		__syncthreads();
	}
	if (tid==0)
	{
		dst_data[blockIdx.x] = a[0];
	}
}

template <int N>
static __global__ void distance_compute_kernel(
	const float *data,
	int          data_pitch,
	int          data_length,
	const float *cluster_centers,
	int          distr_cnt,
	const int   *cluster_id,
	float       *distance)
{
	extern __shared__ float buf[];
	int tid = threadIdx.x;
	int ofs = BLOCK_SIZE1D*blockIdx.x + tid;

	int centers_len = distr_cnt*N;
	float *sm_distance = buf + centers_len;

	for (int i = tid; i < centers_len; i+= BLOCK_SIZE1D)
	{
		int i1 = i / distr_cnt;
		int i2 = i - i1*distr_cnt;
		buf[i] = cluster_centers[i2*N + i1];
	}
	__syncthreads();

	if (ofs < data_length)
	{
		const float *p = data + ofs;
		float d = 0.f;
		float *sm_cluster_centers = buf + cluster_id[ofs];
#pragma unroll
		for (int i = 0; i < N; i++, p+= data_pitch, sm_cluster_centers+= distr_cnt)
		{
			float dif = *p - *sm_cluster_centers;
			d += dif*dif;
		}
		sm_distance[tid] = d;
	}
	else
	{
		sm_distance[tid] = 0.f;
	}
	__syncthreads();

	for (int step = BLOCK_SIZE1D/2; step > 0; step >>= 1)
	{
		if (tid < step)
		{
			float *p_sm = sm_distance + tid;
			p_sm[0] += p_sm[step];
		}
		__syncthreads();
	}
	if (tid==0)
	{
		distance[blockIdx.x] = sm_distance[0];
	}
}

template <int N>
static __global__ void find_farthest_kernel(
	const float *data,
	int          data_pitch,
	int          data_length,
	const float *cluster_centers,
	int          distr_id,
	const int   *cluster_id,
	unsigned long long *out_farthest)
{
	extern __shared__ float buf[];

	unsigned long long *max_item = (unsigned long long *)buf;
	float *sm_centers = buf + sizeof(unsigned long long)/sizeof(float);

	int tid = threadIdx.x;
	int ofs = BLOCK_SIZE1D*blockIdx.x + tid;

	if (tid==0)
	{
		*max_item = 0;
	}
	if (tid < N)
	{
		sm_centers[tid] = cluster_centers[tid];
	}
	__syncthreads();

	if (ofs < data_length && cluster_id[ofs]==distr_id)
	{
		const float *p = data + ofs;

		float distance = 0.f;
#pragma unroll
		for (int j = 0; j < N; j++, p += data_pitch, sm_centers++)
		{
			float delta = *sm_centers - *p;
			distance += delta*delta;
		}

		if (distance > *(float*)&max_item)
		{
			unsigned long long new_max_item = (unsigned int)__float_as_int(distance);
			new_max_item |= ((unsigned long long)(unsigned int)ofs << 32);
			unsigned long long prev_max_item = *max_item;
			for (;;)
			{
				if (distance <= *(float*)max_item)
					break;
				unsigned long long old = atomicCAS(max_item, prev_max_item, new_max_item);
				if (old==prev_max_item)
					break;
				prev_max_item = old;
			}
		}
	}
	__syncthreads();
	if (tid==0)
	{
		out_farthest[blockIdx.x] = *max_item;
	}
}

static __global__ void farthest_reduce_kernel(
	const unsigned long long *in_farthest,
	int cnt,
	unsigned long long *out_farthest)
{
	__shared__ unsigned long long max_item;

	int tid = threadIdx.x;
	int ofs = BLOCK_SIZE1D*blockIdx.x + tid;

	if (tid==0)
	{
		max_item = 0;
	}
	__syncthreads();

	if (ofs < cnt)
	{
		unsigned long long v = in_farthest[ofs];
		float distance = __int_as_float((unsigned int)v);

		if (distance > *(float*)&max_item)
		{
			unsigned long long prev_max_item = max_item;
			for (;;)
			{
				if (distance <= *(float*)&max_item)
					break;
				unsigned long long old = atomicCAS(&max_item, prev_max_item, v);
				if (old==prev_max_item)
					break;
				prev_max_item = old;
			}
		}
	}
	__syncthreads();
	if (tid==0)
	{
		out_farthest[blockIdx.x] = max_item;
	}
}

///
template <int N>
static void lunch_assign_nearest_cluster(
	const float *data,
	int          data_pitch,
	int          data_length,
	const float *cluster_centers,
	int distr_cnt,
	int *cluster)
{
	int block_cnt = (data_length + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;
	int sm_size   = (N*distr_cnt + N*BLOCK_SIZE1D)*sizeof(float);

	assign_nearest_cluster_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(data, data_pitch, cluster_centers, distr_cnt, cluster, data_length);
}

static void reduce_internal(
	reduce_buf &b,
	int width,
	int row_cnt)
{
	while (width > 1)
	{
		float *t = b.buf1;
		b.buf1 = b.buf0;
		b.buf0 = t;
		int block_cnt = (width + 2*BLOCK_SIZE1D - 1) / (2*BLOCK_SIZE1D);

		for (int i = 0; i < row_cnt; i++)
		{
			reduce_kernel<<<block_cnt, BLOCK_SIZE1D>>>(b.buf1+i*b.pitch4, b.buf0+i*b.pitch4, width);
		}
		width = block_cnt;
	}
}

template <int N>
static void lanch_cluster_centers(
	const float *data,
	int          data_pitch,
	const int   *cluster_id,
	int data_length,
	int cluster_cnt,
	reduce_buf &b)
{
	int stat_fld_cnt = N + 1;
	int block_cnt = (data_length + BLOCK_SIZE1D*BLOCK_SIZE1D - 1) / (BLOCK_SIZE1D*BLOCK_SIZE1D);

	int row_cnt = cluster_cnt*stat_fld_cnt;
	int sm_pitch = 32*cluster_cnt;
	int sm_size  = sm_pitch*stat_fld_cnt*sizeof(float);

	cluster_centers_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(
		data,
		data_pitch,
		cluster_id,
		data_length,
		cluster_cnt,
		sm_pitch,
		row_cnt,
		b.buf0,
		b.pitch4);

	reduce_internal(b, block_cnt, row_cnt);
}

template <int N>
static void distance_compute(
	const float *data,
	int          data_pitch,
	int          data_length,
	const float *cluster_centers,
	int          distr_cnt,
	const int   *cluster_id,
	reduce_buf  &distance)
{
	int block_cnt = (data_length + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;
	int sm_size = distr_cnt*N*sizeof(float) + BLOCK_SIZE1D*sizeof(float);

	distance_compute_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(data, data_pitch, data_length, cluster_centers, distr_cnt, cluster_id, distance.buf0);

	reduce_internal(distance, block_cnt, 1);
}

static void farthest_reduce(
	const unsigned long long *in_farthest,
	int data_length,
	unsigned long long *out_farthest)
{
	int block_cnt = (data_length + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;

	if (block_cnt==1)
	{
		farthest_reduce_kernel<<<block_cnt, BLOCK_SIZE1D>>>(in_farthest, data_length, out_farthest);
	}
	else
	{
		unsigned long long *tmp;
		cudaMalloc((void**)&tmp, block_cnt*sizeof(unsigned long long));
		farthest_reduce_kernel<<<block_cnt, BLOCK_SIZE1D>>>(in_farthest, data_length, tmp);
		farthest_reduce(tmp, block_cnt, out_farthest);
		cudaFree(tmp);
	}
}

template <int N>
static void lanch_find_farthest(
	const float *dv_data,
	int          data_pitch,
	int          data_length,
	const float *dv_center,
	int          distr_id,
	const int   *dv_cluster_id,
	unsigned long long *out_farthest)
{
	int block_cnt = (data_length + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;
	int sm_size   = sizeof(unsigned long long) + N*sizeof(float);

	if (block_cnt==1)
	{
		find_farthest_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(dv_data, data_pitch, data_length, dv_center, distr_id, dv_cluster_id, out_farthest);
	}
	else
	{
		unsigned long long *tmp;
		cudaMalloc((void**)&tmp, block_cnt*sizeof(unsigned long long));
		find_farthest_kernel<N><<<block_cnt, BLOCK_SIZE1D, sm_size>>>(dv_data, data_pitch, data_length, dv_center, distr_id, dv_cluster_id, tmp);
		farthest_reduce(tmp, block_cnt, out_farthest);
		cudaFree(tmp);
	}
}

static void find_minmax_element(float *dv_data_, int data_length_, float &min_v, float &max_v)
{
	thrust::device_ptr<float> p(dv_data_);

	max_v = 
		thrust::reduce(p, p+data_length_,
		-FLT_MAX,
		thrust::maximum<float>());

	min_v = 
		thrust::reduce(p, p+data_length_,
		FLT_MAX,
		thrust::minimum<float>());
}

static void alloc_reduce_buf(reduce_buf *b, int capacity, int row_cnt)
{
	cudaMallocPitch((void**)&b->buf0, &b->pitch4, capacity*sizeof(float), row_cnt);
	cudaMallocPitch((void**)&b->buf1, &b->pitch4, capacity*sizeof(float), row_cnt);
	b->pitch4 /= sizeof(float);
	b->capacity = capacity;
	b->row_cnt  = row_cnt;
}

static void free_reduce_buf(reduce_buf *b)
{
	cudaFree(b->buf0);
	cudaFree(b->buf1);
	memset(b, 0, sizeof(reduce_buf));
}

#ifdef TIME_MEASURE
#define	TIME_START()		cudaEventRecord(start_, 0)
#define TIME_STOP(tm)		do {cudaEventRecord(stop_, 0); cudaEventSynchronize(stop_); cudaEventElapsedTime(&tm, start_, stop_);} while (0)
#define TIME_STOP_ADD(tm)	do {float tmp; TIME_STOP(tmp); tm += tmp;} while (0)
#else
#define	TIME_START()
#define TIME_STOP(tm)
#define TIME_STOP_ADD(tm)
#endif

template <int N>
bool k_means(
	float *dv_data_,
	int    data_pitch,
	int   *dv_cluster,
	int    data_length_,
	int    distr_cnt_,
	std::vector<float> &cluster_centers,
	int   attempts_cnt,
	int   max_iter_cnt,
	float epsilon,
	k_means_visualizer &visualizer)
{
#ifdef TIME_MEASURE
	float tm0_all = 0.f, tm1_all = 0.f, tm2_all = 0.f;
	cudaEvent_t start_, stop_;
	cudaEventCreate(&start_, 0);
	cudaEventCreate(&stop_, 0);
#endif

	struct A
	{
		float distance;
		__int32 idx;
	};

	int row_cnt = (N + 1) * distr_cnt_;
	std::vector<float> a(row_cnt);
	float *count = &a[0];
	float *center[N];
	for (int i = 0; i < N; i++)
	{
		center[i] = &a[(i+1)*distr_cnt_];
	}

	int *dv_cluster_tmp = dv_cluster;
	if (attempts_cnt > 1)
	{
		cudaMalloc((void**)&dv_cluster_tmp, data_length_*sizeof(int));
	}
	float *dv_centers;
	cudaMalloc((void**)&dv_centers, distr_cnt_*N*sizeof(float));
	unsigned long long *dv_farthest;
	cudaMalloc((void**)&dv_farthest, sizeof(unsigned long long));

	int max_block_cnt = (data_length_ + BLOCK_SIZE1D - 1) / (BLOCK_SIZE1D);
	reduce_buf b;
	alloc_reduce_buf(&b, max_block_cnt, row_cnt);

	float min_v[N], max_v[N];
	for (int i = 0; i < N; i++)
	{
		find_minmax_element(dv_data_ + i*data_pitch, data_length_, min_v[i], max_v[i]);
	}

	bool interrupted = false;
	float min_distance = FLT_MAX;
	std::vector<float> cluster_centers_tmp(N*distr_cnt_);
	for (int att = 0; att < attempts_cnt && !interrupted; att++)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < distr_cnt_; j++)
			{
				cluster_centers_tmp[j*N + i] = min_v[i] + ((float)rand()/(float)(RAND_MAX+1))*(max_v[i]-min_v[i]);
			}
		}

		int iter = 0;
		bool done = false;
		while (!done && !interrupted)
		{
#ifdef TIME_MEASURE
			float tm0 = 0.f, tm1 = 0.f, tm2 = 0.f;
#endif
			cudaMemcpy(dv_centers, &cluster_centers_tmp[0], distr_cnt_*N*sizeof(float), cudaMemcpyHostToDevice);

			TIME_START();
			lunch_assign_nearest_cluster<N>(dv_data_, data_pitch, data_length_, dv_centers, distr_cnt_, dv_cluster_tmp);
			TIME_STOP(tm0);

			TIME_START();
			lanch_cluster_centers<N>(dv_data_, data_pitch, dv_cluster_tmp, data_length_, distr_cnt_, b);
			TIME_STOP(tm1);

			cudaMemcpy2D(&a[0], sizeof(float), b.buf0, b.pitch4*sizeof(float), sizeof(float), row_cnt, cudaMemcpyDeviceToHost);

			for (int i = 0; i < distr_cnt_; i++)
			{
				if (count[i] != 0)
					continue;

				int max_cnt_idx = 0;
				for (int j = 1; j < distr_cnt_; j++)
				{
					if (count[j] > count[max_cnt_idx])
						max_cnt_idx = j;
				}

				TIME_START();
				lanch_find_farthest<N>(dv_data_, data_pitch, data_length_, dv_centers + N*max_cnt_idx, max_cnt_idx, dv_cluster_tmp, dv_farthest);
				TIME_STOP_ADD(tm2);

				A farthest;
				cudaMemcpy(&farthest, dv_farthest, sizeof(A), cudaMemcpyDeviceToHost);
				float sample[N];
				cudaMemcpy2D(&sample, sizeof(float), dv_data_ + farthest.idx, data_pitch*sizeof(float), sizeof(float), N, cudaMemcpyDeviceToHost);
				cudaMemcpy(dv_cluster_tmp + farthest.idx, &i, sizeof(int), cudaMemcpyHostToDevice);
				count[i]           += 1.f;
				count[max_cnt_idx] -= 1.f;
				for (int j = 0; j < N; j++)
				{
					center[j][i]             += sample[j];
					center[j][max_cnt_idx]   -= sample[j];
				}
			}
			float max_center_shift = 0.f;
			for (int i = 0; i < distr_cnt_; i++)
			{
				float center_shift = 0.f;
				float *c = &cluster_centers_tmp[i*N];
				for (int j = 0; j < N; j++)
				{
					float v = center[j][i] / count[i];
					float t = c[j] - v;
					center_shift += t*t;
					c[j] = v;
				}
				if (center_shift > max_center_shift)
					max_center_shift = center_shift;
			}

#ifdef TIME_MEASURE
			printf("\tassign_nearest_cluster = %f\n", tm0);
			printf("\tcluster_centers        = %f\n", tm1);
			printf("\tfind_farthest          = %f\n", tm2);
			tm0_all += tm0;
			tm1_all += tm1;
			tm2_all += tm2;
#endif
			float distance = FLT_MAX;
			if (++iter==max_iter_cnt || max_center_shift < epsilon*epsilon)
			{
				if (attempts_cnt > 1)
				{
					distance_compute<N>(dv_data_, data_pitch, data_length_, dv_centers, distr_cnt_, dv_cluster_tmp, b);
					cudaMemcpy(&distance, b.buf0, sizeof(float), cudaMemcpyDeviceToHost);
					if (distance < min_distance)
					{
						min_distance = distance;
						cluster_centers = cluster_centers_tmp;
						cudaMemcpy(dv_cluster, dv_cluster_tmp, data_length_*sizeof(int), cudaMemcpyDeviceToDevice);
					}
				}
				done = true;
			}
			visualizer.visualize(cluster_centers, cluster_centers_tmp, N, iter, max_center_shift, min_distance, distance, done, interrupted);
		}
	}

	if (dv_cluster_tmp!=dv_cluster)
	{
		cudaFree(dv_cluster_tmp);
	}
	cudaFree(dv_farthest);
	cudaFree(dv_centers);
	free_reduce_buf(&b);

#ifdef TIME_MEASURE
	printf("assign_nearest_cluster = %f\n", tm0_all);
	printf("cluster_centers        = %f\n", tm1_all);
	printf("find_farthest          = %f\n", tm2_all);
	cudaEventDestroy(start_);
	cudaEventDestroy(stop_);
#endif

	return !interrupted;
}
