#ifndef IMPLICIT_LEARNINGRATE_H
#define IMPLICIT_LEARNINGRATE_H

#include "implicit_basedef.h"
#include "implicit_data.h"

using namespace arma;

struct Imp_Learn_Rate_Base;
struct Imp_Unidim_Learn_Rate;
struct Imp_Unidim_Eigen_Learn_Rate;
struct Imp_Pdim_Learn_Rate;
struct Imp_Pdim_Weighted_Learn_Rate;

struct Imp_Learn_Rate_Base
{
  
#if DEBUG
  virtual ~Imp_Learn_Rate_Base() {
    Rcpp::Rcout << "learning rate object released" << std::endl;
  }
#endif

  virtual mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p) = 0;
};

/* 1 dimension (scalar) learning rate, suggested in Xu's paper
 */
struct Imp_Unidim_Learn_Rate : public Imp_Learn_Rate_Base
{
  Imp_Unidim_Learn_Rate(double g, double a, double c_, double s) :
  gamma(g), alpha(a), c(c_), scale(s) { }
  
  virtual mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p) {
    double lr = scale * gamma * pow(1 + alpha * gamma * t, -c);
    mat lr_mat = mat(p, p, fill::eye) * lr;
    return lr_mat;
  }
private:
  double gamma;
  double alpha;
  double c;
  double scale;
};

// p dimension learning rate
struct Imp_Unidim_Eigen_Learn_Rate : public Imp_Learn_Rate_Base
{
  Imp_Unidim_Eigen_Learn_Rate(const score_func_type& sf) : score_func(sf) { }

  virtual mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p) {
    mat Gi = score_func(theta_old, data_pt, offset);
    mat fisher_est = diagmat(Gi * Gi.t());
    // tr(Fisher_matrix) = sum of eigenvalues of Fisher_matrix
    double sum_eigen = trace(fisher_est);
    // 1 / min_eigen >= p / tr(Fisher_matrix)
    double min_eigen_recpr_lower = p / sum_eigen;

    mat lr_mat = mat(p, p, fill::eye) * min_eigen_recpr_lower / t;
    return lr_mat;
  }

private:
  score_func_type score_func;
};

// p dimension learning rate
struct Imp_Pdim_Learn_Rate : public Imp_Learn_Rate_Base
{
  Imp_Pdim_Learn_Rate(unsigned p, const score_func_type& sf) : 
    Idiag(mat(p, p, fill::eye)), score_func(sf) { }

  virtual mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p) {
    mat Gi = score_func(theta_old, data_pt, offset);
    Idiag = Idiag + diagmat(Gi * Gi.t());
    mat Idiag_inv(Idiag);

    for (unsigned i = 0; i < p; ++i) {
      if (std::abs(Idiag.at(i, i)) > 1e-8) {
        Idiag_inv.at(i, i) = 1. / pow(Idiag.at(i, i), -2./3.);
      }
    }

    return Idiag_inv;
  }

private:
  mat Idiag;
  score_func_type score_func;
};

// p dimension learning rate weighted by alpha
struct Imp_Pdim_Weighted_Learn_Rate : public Imp_Learn_Rate_Base
{
  Imp_Pdim_Weighted_Learn_Rate(unsigned p, double a, const score_func_type& sf) :
    Idiag(mat(p, p, fill::eye)), alpha(a), score_func(sf) { }

  virtual mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset,
                          unsigned t, unsigned p) {
    mat Gi = score_func(theta_old, data_pt, offset);
    Idiag = (1.-alpha) * Idiag + alpha * diagmat(Gi * Gi.t());
    mat Idiag_inv(Idiag);

    for (unsigned i = 0; i < p; ++i) {
      if (std::abs(Idiag.at(i, i)) > 1e-8) {
        Idiag_inv.at(i, i) = 1. / pow(Idiag.at(i, i), -2./3.);
      }
    }

    return Idiag_inv;
  }

private:
  mat Idiag;
  double alpha;
  score_func_type score_func;
};

#endif