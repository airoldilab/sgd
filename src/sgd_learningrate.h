#ifndef SGD_LEARNINGRATE_H
#define SGD_LEARNINGRATE_H

#include "sgd_basedef.h"
#include "sgd_data.h"

using namespace arma;

struct Sgd_Learn_Rate_Base;
struct Sgd_Onedim_Learn_Rate;
struct Sgd_Onedim_Eigen_Learn_Rate;
struct Sgd_Ddim_Learn_Rate;

struct Sgd_Learn_Rate_Base
{

#if DEBUG
  virtual ~Sgd_Learn_Rate_Base() {
    Rcpp::Rcout << "Learning rate object released" << std::endl;
  }
#endif
  virtual ~Sgd_Learn_Rate_Base() {}
  virtual mat learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset,
                          unsigned t, unsigned d) = 0;
};

/* one-dimensional (scalar) learning rate, suggested in Xu's paper
 */
struct Sgd_Onedim_Learn_Rate : public Sgd_Learn_Rate_Base
{
  Sgd_Onedim_Learn_Rate(double g, double a, double c_, double s) :
  gamma(g), alpha(a), c(c_), scale(s) { }

  virtual mat learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset,
                          unsigned t, unsigned d) {
    double lr = scale * gamma * pow(1 + alpha * gamma * t, -c);
    mat lr_mat = mat(d, d, fill::eye) * lr;
    return lr_mat;
  }

private:
  double gamma;
  double alpha;
  double c;
  double scale;
};

// one-dimensional learning rate to parameterize a diagonal matrix
struct Sgd_Onedim_Eigen_Learn_Rate : public Sgd_Learn_Rate_Base
{
  Sgd_Onedim_Eigen_Learn_Rate(const score_func_type& sf) : score_func(sf) { }

  virtual mat learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset,
                          unsigned t, unsigned d) {
    mat Gi = score_func(theta_old, data_pt, offset);
    mat fisher_est = diagmat(Gi * Gi.t());
    // tr(Fisher_matrix) = sum of eigenvalues of Fisher_matrix
    double sum_eigen = trace(fisher_est);
    // min_eigen <= d / tr(Fisher_matrix)
    double min_eigen_recpr_lower = d / sum_eigen;

    mat lr_mat = mat(d, d, fill::eye) * min_eigen_recpr_lower / t;
    return lr_mat;
  }

private:
  score_func_type score_func;
};

// d-dimensional learning rate with parameter weight alpha and exponent c
// AdaGrad: special case where alpha=0, c=1/2
// d-dim: special case where alpha=1, c=1
struct Sgd_Ddim_Learn_Rate : public Sgd_Learn_Rate_Base
{
  Sgd_Ddim_Learn_Rate(unsigned d, double a, double c_, const score_func_type& sf) :
    Idiag(mat(d, d, fill::eye)), alpha(a), c(c_), score_func(sf) { }

  virtual mat learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset,
                          unsigned t, unsigned d) {
    mat Gi = score_func(theta_old, data_pt, offset);
    Idiag = (1.-alpha) * Idiag + alpha * diagmat(Gi * Gi.t());
    mat Idiag_inv(Idiag);

    for (unsigned i = 0; i < d; ++i) {
      if (std::abs(Idiag.at(i, i)) > 1e-8) {
        Idiag_inv.at(i, i) = 1. / pow(Idiag.at(i, i), c);
      }
    }
    return Idiag_inv;
  }


private:
  mat Idiag;
  double alpha;
  double c;
  score_func_type score_func;
};

#endif
