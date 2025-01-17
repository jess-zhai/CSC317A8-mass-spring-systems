#include "fast_mass_springs_step_dense.h"
#include <igl/matlab_format.h>

void fast_mass_springs_step_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::MatrixXd & M,
  const Eigen::MatrixXd & A,
  const Eigen::MatrixXd & C,
  const Eigen::LLT<Eigen::MatrixXd> & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  //////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  Eigen::MatrixXd y = (1/pow(delta_t, 2) * (M * (2 * Ucur - Uprev))) + fext;
  Unext = Ucur;
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(E.rows(), 3);
  for(int iter = 0;iter < 50;iter++)
  {
    for (int i = 0; i < E.rows(); i++) {
      d.row(i) = (Unext.row(E(i, 0)) - Unext.row(E(i, 1))).normalized() * r(i);
    }
    Eigen::MatrixXd l = k * A.transpose() * d + y + 1e7 * C.transpose() * C * V;
    Unext = prefactorization.solve(l);
  }
  //////////////////////////////////////////////////////////////////////////////
}
