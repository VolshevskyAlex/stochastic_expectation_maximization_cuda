#pragma once

#include "em.h"

template <int channels>
void fill_data4simulation(
	float *dv_data,
	size_t pitch4,
	int len,
	int distr_cnt,
	int rnd_seed,
	double min_sigma,
	double max_sigma,
	std::vector<distr_gauss_params<channels> > &gmm);
