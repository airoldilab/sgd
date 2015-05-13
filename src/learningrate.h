#ifndef LEARNINGRATE_H
#define LEARNINGRATE_H

#include "basedef.h"
#include "data.h"

using namespace arma;

struct Sgd_Learn_Rate_Base;
struct Sgd_Onedim_Learn_Rate;
struct Sgd_Onedim_Eigen_Learn_Rate;
struct Sgd_Ddim_Learn_Rate;
struct Sgd_Learn_Rate_Value;



// The return value for learning_rate method
struct Sgd_Learn_Rate_Value
{

  Sgd_Learn_Rate_Value(unsigned t, unsigned d): type(t), dim(d) {
    if (type == 0) lr_scalar = 1;
    else lr_mat = mat(d, d, fill::eye);
  }

  mat lr_mat;
  double lr_scalar;
  unsigned type;  // type: 0 for scalar; 1 for mat
  unsigned dim;
};

mat operator*(const Sgd_Learn_Rate_Value& lr, const mat& score){
  if (lr.type == 0) return lr.lr_scalar * score;
  else return lr.lr_mat * score;
}

std::ostream& operator<<(std::ostream& os, const Sgd_Learn_Rate_Value& lr){
  if (lr.type == 0) os << lr.lr_scalar;
  else os << lr.lr_mat;
  return os;
}


struct Sgd_Learn_Rate_Base
{
#if DEBUG
  virtual ~Sgd_Learn_Rate_Base() {
    Rcpp::Rcout << "Learning rate object released" << std::endl;
  }
#else
  virtual ~Sgd_Learn_Rate_Base() {}
#endif
  virtual const Sgd_Learn_Rate_Value& learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt,
                            double offset, unsigned t, unsigned d) = 0;
};

/* one-dimensional (scalar) learning rate, suggested in Xu's paper
 */
struct Sgd_Onedim_Learn_Rate : public Sgd_Learn_Rate_Base
{
  Sgd_Onedim_Learn_Rate(double g, double a, double c_, double s) :
  gamma(g), alpha(a), c(c_), scale(s), v(0, 1) { }

  virtual const Sgd_Learn_Rate_Value& learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt,
                            double offset, unsigned t, unsigned d) {
    v.lr_scalar = scale * gamma * pow(1 + alpha * gamma * t, -c);
    return v;
  }

private:
  double gamma;
  double alpha;
  double c;
  double scale;
  Sgd_Learn_Rate_Value v;
};

// one-dimensional learning rate to parameterize a diagonal matrix
struct Sgd_Onedim_Eigen_Learn_Rate : public Sgd_Learn_Rate_Base
{
  Sgd_Onedim_Eigen_Learn_Rate(const score_func_type& sf) : score_func(sf), v(0, 1) { }

  virtual const Sgd_Learn_Rate_Value& learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt,
                            double offset, unsigned t, unsigned d) {
    mat Gi = score_func(theta_old, data_pt, offset);
    // tr(Fisher_matrix) = sum of eigenvalues of Fisher_matrix
    //mat fisher_est = diagmat(Gi * Gi.t()); // vectorized version
    //double sum_eigen = trace(fisher_est);
    double sum_eigen = 0;
    for (unsigned i = 0; i < d; ++i) {
      sum_eigen += pow(Gi.at(i, 0), 2);
    }
    // min_eigen <= d / trace(Fisher_matrix)
    double min_eigen_upper = sum_eigen / d;
    v.lr_scalar = 1. / (min_eigen_upper * t);
    return v;
  }

private:
  score_func_type score_func;
  Sgd_Learn_Rate_Value v;
};

// d-dimensional learning rate with parameter weight alpha and exponent c
// AdaGrad: alpha=1, c=1/2
// d-dim: alpha=0, c=1
struct Sgd_Ddim_Learn_Rate : public Sgd_Learn_Rate_Base
{
  Sgd_Ddim_Learn_Rate(unsigned d, double a, double c_, const score_func_type& sf) :
    Idiag(mat(d, d, fill::eye)), alpha(a), c(c_), score_func(sf), v(1, d) { }

  virtual const Sgd_Learn_Rate_Value& learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt,
                            double offset, unsigned t, unsigned d) {
    mat Gi = score_func(theta_old, data_pt, offset);
    //Idiag = alpha * Idiag + diagmat(Gi * Gi.t()); // vectorized version
    for (unsigned i = 0; i < d; ++i) {
      Idiag.at(i, i) = alpha * Idiag.at(i, i) + pow(Gi.at(i, 0), 2);
    }

    // mat Idiag_inv(Idiag);
    // for (unsigned i = 0; i < d; ++i) {
    //   if (std::abs(Idiag.at(i, i)) > 1e-8) {
    //     Idiag_inv.at(i, i) = 1. / pow(Idiag.at(i, i), c);
    //   }
    // }
    // v.lr_mat = Idiag_inv;
    for (unsigned i = 0; i < d; ++i) {
      if (std::abs(Idiag.at(i, i)) > 1e-8) {
        v.lr_mat.at(i, i) = 1. / pow(Idiag.at(i, i), c);
      }
      else{
        v.lr_mat.at(i, i) = Idiag.at(i, i); 
      }
    }
    return v;
  }

private:
  mat Idiag;
  double alpha;
  double c;
  score_func_type score_func;
  Sgd_Learn_Rate_Value v;
};





#endif
