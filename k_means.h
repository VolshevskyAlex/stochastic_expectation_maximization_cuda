#pragma once

#include <vector>

class k_means_visualizer
{
public:
	virtual void visualize(
		const std::vector<float> &cluster_centers_best,
		const std::vector<float> &cluster_centers,
		int Ndim,
		int iter,
		float current_shift,
		float min_distance,
		float distance,
		bool attempt_done,
		bool &interrupt)
	{
	}
};

template <int N>
bool k_means(
	float *dv_data_,
	int    data_pitch,
	int   *dv_cluster_,
	int    data_length_,
	int    distr_cnt_,
	std::vector<float> &cluster_centers,
	int   attempts_cnt,
	int   max_iter_cnt,
	float epsilon,
	k_means_visualizer &visualizer);
