#pragma once

#include <Windows.h>
#include <Eigen/SVD>
#include <vector>
#include <memory>

#include "em.h"
#include "render_tables.h"

class Shape
{
public:
	virtual ~Shape() {}
	virtual void Render() = 0;
	virtual int GetType() = 0;
};

typedef std::vector<std::unique_ptr<Shape> > vec_shapes;

class Shpere : public Shape
{
	int type_;
	float x_, y_, z_;
public:
	Shpere(float x, float y, float z, int type) :
		x_(x), y_(y), z_(z), type_(type)
	{
	}
	void Render();
	int GetType()
	{
		return type_;
	}
};

class Ellipsoid : public Shape
{
	typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3;
	double matr_[16];
	float  color_[3];

public:
	Ellipsoid(const double *mu, const double *c, const float *color);
	void Render();
	int GetType()
	{
		return 2;
	}
};

class Simulation;

class k_means_visualizer3 : public k_means_visualizer
{
	Simulation *sim_;

public:
	void set_owner(Simulation *s)
	{
		sim_ = s;
	}
	virtual void visualize(
		const std::vector<float> &cluster_centers_best,
		const std::vector<float> &cluster_centers,
		int Ndim,
		int iter,
		float current_shift,
		float min_distance,
		float distance,
		bool attempt_done,
		bool &terminate);
};

class em_visualizer3 : public em_visualizer<3>
{
	Simulation *sim_;

public:
	void set_owner(Simulation *s)
	{
		sim_ = s;
	}
	virtual void visualize(
		const std::vector<distr_gauss_params<3> > &gmm,
		int iter,
		float log_likelihood,
		bool &terminate);
};

class Simulation
{
	enum {req_empty, req_start, req_stop, req_term, req_continue};
	bool is_run_;
	bool iter_completed_, att_completed_;
	int  req_;
	HANDLE thread_;
	CRITICAL_SECTION   cs_req_, cs_resp_, cs_shape_list_, cs_distr_view_;
	CONDITION_VARIABLE cv_req_, cv_resp_;

	vec_shapes              shape_list_;
	distr_view              distr_view_;
	k_means_visualizer3     vis_k_means_;
	em_visualizer3          vis_em_;

	void main_loop();
	void terminate();
	static DWORD WINAPI thread_fn(void*);

public:
	float *dv_data_;
	int    data_pitch_;
	int    data_len_;
	int    cluster_cnt_;
	int    rand_seed_;
	std::vector<distr_gauss_params<3> > gmm_;
	std::vector<float> distr_colors_;

	Simulation(int data_len);
	~Simulation();

	void start();
	void stop();
	bool is_run();
	bool is_iteration_completed();
	bool is_att_completed();
	void sim_continue();

	vec_shapes &lock_shape_list();
	void unlock_shape_list();

	distr_view &lock_distr_view();
	void unlock_distr_view();

// вызывается из visualizer
	void vis_signal_and_wait(bool att_completed, bool &interrupt);
};
