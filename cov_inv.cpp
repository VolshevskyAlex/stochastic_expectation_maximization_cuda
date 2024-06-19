#include "cov_inv.h"

#include <Eigen/SVD>

void calc_inverse_cov(double *c, double *cov_i, double &det, int N)
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

	const double eps = 1e-12;
	Mat cov(N, N);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			cov(i, j) = c[i*N+j];

	Eigen::JacobiSVD<Mat> svd(cov, Eigen::ComputeThinV);
	Mat V = svd.matrixV();
	Mat sv = svd.singularValues();
	Mat S(N, N);
	det = 1.0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (i==j)
			{
				double v = std::max(sv(i, 0), eps);
				det *= v;
				S(i, j) = 1.0 / v;
			}
			else
				S(i, j) = 0.0;

	Mat t = V*S*V.transpose();

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			cov_i[i*N+j] = t(i, j);
}
