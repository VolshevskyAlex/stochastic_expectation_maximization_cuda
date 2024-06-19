#define TIME_MEASURE

#include "em_impl.h"

template
void expectation_maximization<3>(
	float *dv_data,
	int data_pitch,
	int data_length,
	int distr_cnt,
	std::vector<distr_gauss_params<3> > &d,
	k_means_visualizer &v1,
	em_visualizer<3>   &v2);
