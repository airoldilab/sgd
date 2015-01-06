#ifndef IMPLICIT_LEARNINGRATE_HPP
#define IMPLICIT_LEARNINGRATE_HPP

#include "implicit_basedef.hpp"
#include "implicit_data.hpp"

using namespace arma;

struct Imp_Unidim_Learn_Rate;
struct Imp_Pxdim_Learn_Rate;

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
struct Imp_Pxdim_Learn_Rate
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

mat Imp_Pxdim_Learn_Rate::Idiag = mat();

#endif