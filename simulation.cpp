#include "simulation.h"

#include <cuda_runtime.h>
#include <glew.h>
#include <freeglut.h>

Simulation::Simulation(int len)
{
	vis_k_means_.set_owner(this);
	vis_em_.set_owner(this);

	InitializeCriticalSection(&cs_req_);
	InitializeCriticalSection(&cs_resp_);
	InitializeCriticalSection(&cs_shape_list_);
	InitializeCriticalSection(&cs_distr_view_);
	InitializeConditionVariable(&cv_req_);
	InitializeConditionVariable(&cv_resp_);

	is_run_         = false;
	iter_completed_ = false;
	att_completed_  = false;
	req_            = req_empty;

	rand_seed_ = 0;
	data_len_ = len;
	size_t pitch;
	cudaMallocPitch((void**)&dv_data_, &pitch, data_len_*sizeof(float), 3);
	data_pitch_ = pitch / sizeof(float);

	CreateThread(NULL, 0, thread_fn, this, 0, NULL);
}

Simulation::~Simulation()
{
	terminate();
	cudaFree(dv_data_);
	DeleteCriticalSection(&cs_req_);
	DeleteCriticalSection(&cs_resp_);
	DeleteCriticalSection(&cs_shape_list_);
	DeleteCriticalSection(&cs_distr_view_);
}

DWORD WINAPI Simulation::thread_fn(void *t)
{
	((Simulation*)t)->main_loop();
	return 0;
}

void Simulation::main_loop()
{
	bool done = false;
	int r;
	while (!done)
	{
		EnterCriticalSection(&cs_req_);
		while (req_==req_empty)
		{
			SleepConditionVariableCS(&cv_req_, &cs_req_, INFINITE);
		}
		r = req_;
		req_ = req_empty;
		LeaveCriticalSection(&cs_req_);

		switch (r)
		{
		case req_term:
			done=true;
			break;
		case req_start:
			EnterCriticalSection(&cs_resp_);
			is_run_ = true;
			LeaveCriticalSection(&cs_resp_);
			WakeConditionVariable(&cv_resp_);

			srand(rand_seed_);
			expectation_maximization<3>(dv_data_, data_pitch_, data_len_, cluster_cnt_, gmm_, vis_k_means_, vis_em_);
			EnterCriticalSection(&cs_resp_);
			is_run_ = false;
			iter_completed_ = false;
			att_completed_  = false;
			LeaveCriticalSection(&cs_resp_);
			break;
		}
		WakeConditionVariable(&cv_resp_);
	}
}

void Simulation::start()
{
	EnterCriticalSection(&cs_req_);
	req_  = req_start;
	LeaveCriticalSection(&cs_req_);
	WakeConditionVariable(&cv_req_);

	EnterCriticalSection(&cs_resp_);
	while (!is_run_)
	{
		SleepConditionVariableCS(&cv_resp_, &cs_resp_, INFINITE);
	}
	LeaveCriticalSection(&cs_resp_);
}

void Simulation::stop()
{
	EnterCriticalSection(&cs_req_);
	req_  = req_stop;
	LeaveCriticalSection(&cs_req_);
	WakeConditionVariable(&cv_req_);

	EnterCriticalSection(&cs_resp_);
	while (is_run_)
	{
		SleepConditionVariableCS(&cv_resp_, &cs_resp_, INFINITE);
	}
	LeaveCriticalSection(&cs_resp_);
}

void Simulation::terminate()
{
	EnterCriticalSection(&cs_req_);
	req_  = req_term;
	LeaveCriticalSection(&cs_req_);
	WakeConditionVariable(&cv_req_);

	WaitForSingleObject(thread_, INFINITE);
}

bool Simulation::is_run()
{
	return is_run_;
}

bool Simulation::is_iteration_completed()
{
	return iter_completed_;
}

bool Simulation::is_att_completed()
{
	return att_completed_;
}

void Simulation::sim_continue()
{
	EnterCriticalSection(&cs_resp_);
	att_completed_  = false;
	iter_completed_ = false;
	LeaveCriticalSection(&cs_resp_);

	EnterCriticalSection(&cs_req_);
	req_  = req_continue;
	LeaveCriticalSection(&cs_req_);
	WakeConditionVariable(&cv_req_);
}

void Simulation::vis_signal_and_wait(bool attempt_done, bool &interrupt)
{
	EnterCriticalSection(&cs_resp_);
	iter_completed_ = true;
	att_completed_ = attempt_done;
	LeaveCriticalSection(&cs_resp_);
	WakeConditionVariable(&cv_resp_);

	EnterCriticalSection(&cs_req_);
	while (req_==req_empty)
	{
		SleepConditionVariableCS(&cv_req_, &cs_req_, INFINITE);
	}
	switch (req_)
	{
	case req_stop:
	case req_term:
		interrupt = true;
		break;
	default:
		req_ = req_empty;
	}
	LeaveCriticalSection(&cs_req_);
}

vec_shapes &Simulation::lock_shape_list()
{
	EnterCriticalSection(&cs_shape_list_);
	return shape_list_;
}

void Simulation::unlock_shape_list()
{
	LeaveCriticalSection(&cs_shape_list_);
}

distr_view &Simulation::lock_distr_view()
{
	EnterCriticalSection(&cs_distr_view_);
	return distr_view_;
}

void Simulation::unlock_distr_view()
{
	LeaveCriticalSection(&cs_distr_view_);
}

void k_means_visualizer3::visualize(
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
	vec_shapes &v = sim_->lock_shape_list();
	v.clear();
	for (size_t i = 0; i < cluster_centers_best.size(); i+= 3)
	{
		const float *p = cluster_centers_best.data() + i;
		v.push_back(std::unique_ptr<Shape>(new Shpere(p[0], p[1], p[2], 1)));
	}
	for (size_t i = 0; i < cluster_centers.size(); i+= 3)
	{
		const float *p = cluster_centers.data() + i;
		v.push_back(std::unique_ptr<Shape>(new Shpere(p[0], p[1], p[2], 0)));
	}
	sim_->unlock_shape_list();

	sim_->vis_signal_and_wait(attempt_done, interrupt);
}

void em_visualizer3::visualize(
	const std::vector<distr_gauss_params<3> > &gmm,
	int iter,
	float log_likelihood,	// логарифм правдоподобия
	bool &interrupt)
{
	printf("iteration = %d, likelihood = %f\n", iter, expf(log_likelihood));

	const float *colors = sim_->distr_colors_.data();
	vec_shapes &v = sim_->lock_shape_list();
	v.clear();
	for (size_t i = 0; i < gmm.size(); i++)
	{
		const distr_gauss_params<3> &it = gmm[i];
		v.push_back(std::unique_ptr<Shape>(new Ellipsoid(it.mu, it.cov, colors + i*3)));
	}
	sim_->unlock_shape_list();

	distr_view &view = sim_->lock_distr_view();
	view.Update(gmm, colors);
	sim_->unlock_distr_view();

	sim_->vis_signal_and_wait(false, interrupt);
}

void Shpere::Render()
{
	if (type_==0)
		glColor3f(1.f, 1.f, 1.f);
	else
		glColor3f(0.f, 0.f, 1.f);

	glPushMatrix();
	glTranslatef(x_, y_, z_);
	glutWireSphere(type_==0 ? 0.1 : 0.12, 20, 20);
	glPopMatrix();
}

Ellipsoid::Ellipsoid(const double *mu, const double *c, const float *color)
{
	Mat3 cov(c);
	Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic , Eigen::Dynamic, Eigen::RowMajor> > svd(cov, Eigen::ComputeThinV);
	Mat3 V = svd.matrixV();
	Eigen::Matrix<double, 3, 1> sv = svd.singularValues();
	Mat3 S;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			S(i, j) = i==j ? sqrt(sv[i]) : 0.0;
		}

	Mat3 t = S*V.transpose();

	double a[16] = {
		t(1, 0), t(1, 1), t(1, 2), 0.0,
		t(2, 0), t(2, 1), t(2, 2), 0.0,
		t(0, 0), t(0, 1), t(0, 2), 0.0,
		mu[0]  , mu[1]  , mu[2]  , 1.0
	};
	memcpy(matr_, a, sizeof(matr_));
	memcpy(color_, color, sizeof(color_));
}

void Ellipsoid::Render()
{
	glColor3fv(color_);

	glPushMatrix();
	glMultMatrixd(matr_);
	glutWireSphere(3.0, 20, 20);
	glPopMatrix();
}
