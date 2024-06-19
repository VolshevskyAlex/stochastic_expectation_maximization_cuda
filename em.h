#pragma once

#include <vector>

#include "k_means.h"

template <int N>
struct distr_gauss_params
{
	double tau;
	double mu[N];
	double cov[N*N];
	double cov_inv[N*N];
	double cov_determinant;
};

template <int N>
class em_visualizer
{
public:
	virtual void visualize(
		const std::vector<distr_gauss_params<N> > &gmm,
		int iter,
		float log_likelihood,
		bool &interrupt)
	{
	}
};

template <int N>
void expectation_maximization(
	float *dv_data,
	int data_pitch,
	int data_length,
	int distr_cnt,
	std::vector<distr_gauss_params<N> > &gmm,
	k_means_visualizer &v1,
	em_visualizer<N>   &v2);
