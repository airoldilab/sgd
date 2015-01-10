#ifndef IMPLICIT_LEARNINGRATE_H
#define IMPLICIT_LEARNINGRATE_H

#include "implicit_basedef.h"
#include "implicit_data.h"

using namespace arma;

struct Imp_Unidim_Learn_Rate;
struct Imp_Unidim_Eigen_Learn_Rate;
struct Imp_Pdim_Learn_Rate;
struct Imp_Pdim_Weighted_Learn_Rate;

/* 1 dimension (scalar) learning rate, suggested in Xu's paper
 */
struct Imp_Unidim_Learn_Rate
{
  static mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p,
                          double gamma, double alpha, double c, double scale) {
    double lr = scale * gamma * pow(1 + alpha * gamma * t, -c);
    mat lr_mat = mat(p, p, fill::eye) * lr;
    return lr_mat;
  }
};

// p dimension learning rate
struct Imp_Unidim_Eigen_Learn_Rate
{
  static mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p,
                          score_func_type score_func) {
    mat Gi = score_func(theta_old, data_pt, offset);
    mat fisher_est = diagmat(Gi * Gi.t());
    // tr(Fisher_matrix) = sum of eigenvalues of Fisher_matrix
    double sum_eigen = trace(fisher_est);
    // 1 / min_eigen >= p / tr(Fisher_matrix)
    double min_eigen_recpr_lower = p / sum_eigen;

    mat lr_mat = mat(p, p, fill::eye) * min_eigen_recpr_lower / t;
    return lr_mat;
  }
};

// p dimension learning rate
struct Imp_Pdim_Learn_Rate
{
  static mat Idiag;

  static mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p,
                          score_func_type score_func) {
    mat Gi = score_func(theta_old, data_pt, offset);
    Idiag = Idiag + diagmat(Gi * Gi.t());
    mat Idiag_inv(Idiag);

    for (unsigned i = 0; i < p; ++i) {
      if (abs(Idiag_inv.at(i, i)) > 1e-8) {
        Idiag_inv.at(i, i) = 1. / Idiag_inv.at(i, i);
      }
    }
    return Idiag_inv;
  }

  static void reinit(unsigned p) {
    Idiag = mat(p, p, fill::eye);
  }
};

mat Imp_Pdim_Learn_Rate::Idiag = mat();

// p dimension learning rate weighted by alpha
struct Imp_Pdim_Weighted_Learn_Rate
{
  static mat Idiag;

  static mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p,
                          score_func_type score_func, double alpha) {
    mat Gi = score_func(theta_old, data_pt, offset);
    Idiag = (1.-alpha) * Idiag + alpha * diagmat(Gi * Gi.t());
    mat Idiag_inv(Idiag);

    for (unsigned i = 0; i < p; ++i) {
      if (abs(Idiag_inv.at(i, i)) > 1e-8) {
        Idiag_inv.at(i, i) = 1. / Idiag_inv.at(i, i) / t;
      }
    }

    return Idiag_inv;
  }

  static void reinit(unsigned p) {
    Idiag = mat(p, p, fill::eye);
  }
};

mat Imp_Pdim_Weighted_Learn_Rate::Idiag = mat();


#endif