#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  std::vector<Eigen::Triplet<double> > ijv;
  const int n = V.rows();
  Eigen::SparseMatrix<double> Q(n,n);

  M.resize(n, n);
  for (int i=0; i < n; i++){
    ijv.emplace_back(i,i,m[i]);
  }
  M.setFromTriplets(ijv.begin(), ijv.end());
  ijv.clear();
  r.resize(E.rows());
  for (int i = 0; i < E.rows(); i ++) {
      r[i] = (V.row(E(i, 0)) - V.row(E(i, 1))).norm();
  }
  signed_incidence_matrix_sparse(n, E, A);
  C.resize(b.size(), n);
  for (int i = 0; i < b.rows(); i ++) {
      ijv.emplace_back(i,b[i],1);
  }
  C.setFromTriplets(ijv.begin(), ijv.end());
  ijv.clear();

  Q = k * A.transpose() * A + 1/pow(delta_t, 2) * M + 1e7 * C.transpose() * C;
 
  /////////////////////////////////////////////////////////////////////////////
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
