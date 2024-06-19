#include <Eigen/Dense>
#include <Eigen/cholesky>

#include <cuda_runtime.h>

#include "em.h"

#define _USE_MATH_DEFINES
#include <math.h>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

static void rand_cov(Mat &cov, int n, double min_sigma, double max_sigma)
{
	Mat K;
	K.resize(n, n);

	do {
		for (int i = 0; i < n; i++)
		{
			K(i, i) = 1;
			for (int j = i+1; j < n; j++)
			{
				double t = 0.8*(double)rand() / (RAND_MAX+1);
				K(i, j) = t;
				K(j, i) = t;
			}
		}
	} while (K.determinant() <= 0);

	Mat sigma(1, n);
	for (int i = 0; i < n; i++)
	{
		sigma(i) = min_sigma + (max_sigma-min_sigma)*(double)rand() / (RAND_MAX+1);
	}
	cov.resize(n, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cov(i, j) = K(i, j) * sqrt(sigma(i)*sigma(j));
		}
	}
}

template <int N>
void generate_random_gmm(
	std::vector<distr_gauss_params<N> > &gm,
	int distr_cnt,
	double min_sigma,
	double max_sigma)
{
	std::vector<double> v;
	double v_sum = 0;
	for (int i = 0; i < distr_cnt; i++)
	{
		double a = 0.2 + (double)rand() / (RAND_MAX+1);
		v.push_back(a);
		v_sum += a;
	}

	gm.resize(distr_cnt);
	for (int i = 0; i < distr_cnt; i++)
	{
		distr_gauss_params<N> &item = gm[i];
		item.tau  = (float)(v[i] / v_sum);
		for (int j = 0; j < N; j++)
		{
			item.mu[j] = (float)(2.0 * (double)rand() / (RAND_MAX+1) - 1.0);
		}
		Mat cov;
		rand_cov(cov, N, min_sigma, max_sigma);
		double *cov_p = cov.data();
		for (int j = 0; j < N*N; j++)
		{
			item.cov[j] = (float)cov_p[j];
		}
	}
}

template <int N>
void fill_data_gmm(float *data, int data_pitch, int len, std::vector<std::vector<float> > &gmm);

#if 1
template <int N>
void generate_gmm(float *data, int data_pitch, int count, std::vector<distr_gauss_params<N> > &d)
{
	typedef Eigen::Matrix<double, N, N, Eigen::RowMajor> MatN;

	int distr_params_size = 1 + N + N*(N+1)/2;
	int distr_cnt = (int)d.size();
	std::vector<std::vector<float> > params(distr_cnt);
	for (int i = 0; i < distr_cnt; i++)
	{
		params[i].resize(distr_params_size);
		float *a = params[i].data();
		*a++ = (float)d[i].tau;
		Mat t = Mat(MatN(d[i].cov)).llt().matrixL();
		for (int j = 0; j < N; j++)
		{
			*a++ = (float)d[i].mu[j];
			for (int k = 0; k <= j; k++)
			{
				*a++ = (float)t(j, k);
			}
		}
	}
	fill_data_gmm<N>(data, data_pitch, count, params);
}

template <int N>
void fill_data4simulation(
	float *dv_data,
	size_t pitch4,
	int len,
	int distr_cnt,
	int rnd_seed,
	double min_sigma,
	double max_sigma,
	std::vector<distr_gauss_params<N> > &gmm)
{
	srand(rnd_seed);
	generate_random_gmm<N>(gmm, distr_cnt, min_sigma, max_sigma);

	generate_gmm<N>(dv_data, pitch4, len, gmm);
}

#else
double gen_norm()
{
	static double z[2];
	static int id = 1;
	if (id)
	{
		double x, y, s;
		do {
			x = 2.0*(double)rand() / (RAND_MAX+1) - 1.0;
			y = 2.0*(double)rand() / (RAND_MAX+1) - 1.0;
			s = x*x + y*y;
		}
		while (s > 1.0 || s==0);
		z[0] = x*sqrt(-2.0*log(s)/s);
		z[1] = y*sqrt(-2.0*log(s)/s);
	}
	id ^= 1;
	return z[id];
}


template <int N>
void generate_gmm(float *p_dst, int data_pitch, int count, std::vector<distr_gauss_params<N> > &d)
{
	typedef Eigen::Matrix<double, N, N, Eigen::RowMajor> MatN;

	int distr_cnt = (int)d.size();
	std::vector<std::vector<double> > sigma_ll(distr_cnt);
	for (int i = 0; i < distr_cnt; i++)
	{
		Mat t = Mat(MatN(d[i].cov)).llt().matrixL();
		sigma_ll[i].resize(N*(N+1)/2);
		double *a = sigma_ll[i].data();
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k <= j; k++)
			{
				*a++ = t(j, k);
			}
		}
	}

	for (int i = 0; i < count; i++)
	{
		double p = (double)rand() / (RAND_MAX+1);
		int claster_id = 0;

		while (p > d[claster_id].tau)
		{
			p -= d[claster_id++].tau;
		}
		if (claster_id>=distr_cnt)
		{
			int c = 0;
		}

		double z[N];
		for (int j = 0; j < N; j++)
		{
			z[j] = gen_norm();
		}
		double *mu = d[claster_id].mu;
		double *a  = sigma_ll[claster_id].data();

		float *p_x = p_dst + i;
		for (int j = 0; j < N; j++, p_x += data_pitch)
		{
			double v = mu[j];
			for (int k = 0; k <= j; k++, a++)
			{
				v += *a * z[k];
			}
			*p_x = (float)v;
		}
	}
}
template <int channels>
void fill_data4simulation(
	float *dv_data,
	size_t pitch4,
	int len,
	int distr_cnt,
	int rnd_seed,
	double min_sigma,
	double max_sigma,
	std::vector<distr_gauss_params<channels> > &gmm)
{
	srand(rnd_seed);
	generate_random_gmm<channels>(gmm, distr_cnt, min_sigma, max_sigma);

	float *x = new float[len*3];
	generate_gmm<channels>(x, len, len, gmm);

	cudaMemcpy2D(dv_data, pitch4*sizeof(float), x, len*sizeof(float), len*sizeof(float), channels, cudaMemcpyHostToDevice);
	delete[] x;
}
#endif

template
void fill_data4simulation<3>(
	float *dv_data,
	size_t pitch4,
	int len,
	int distr_cnt,
	int rnd_seed,
	double min_sigma,
	double max_sigma,
	std::vector<distr_gauss_params<3> > &gmm);

/*
void cholesky(cv::Mat1d a, cv::Mat1d &b)
{
	b.create(a.size());
	int n = a.cols;
	double *p_a = a.ptr<double>();
	double *p_b = b.ptr<double>();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (j > i)
			{
				b(i, j) = 0;
			}
			else if (i==j)
			{
				double s = 0;
				for (int k = 0; k < j; k++)
				{
					double t = b(j, k);
					s += t*t;
				}
				b(j, j) = sqrt(a(j, j) - s);
			}
			else
			{
				double s = 0;
				for (int k = 0; k < j; k++)
				{
					s += b(i, k)*b(j, k);
				}
				b(i, j) = (a(i, j) - s) / b(j, j);
			}
		}
	}
	double d = cv::determinant(a);
	cv::Mat c = b*b.t();
}
*/
