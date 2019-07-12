#ifndef MODEL_BASE_MODEL_H
#define MODEL_BASE_MODEL_H

#include "../basedef.h"
#include "../data/data_point.h"

class base_model {
  /**
   * Base class for models
   *
   * @param model attributes affiliated with model as R type
   */
public:
  base_model(Rcpp::List model) {
    name_ = Rcpp::as<std::string>(model["name"]);
    lambda1_ = Rcpp::as<double>(model["lambda1"]);
    lambda2_ = Rcpp::as<double>(model["lambda2"]);
  }

  std::string name() const {
    return name_;
  }

  mat gradient(unsigned t, const mat& theta_old, const data_set& data) const;
  mat gradient_penalty(const mat& theta) const {
    return lambda1_*sign(theta) + lambda2_*theta;
  }

  // Functions for implicit update
  // Following the JSS paper, we assume C_n = identity, lambda = 1, and use ksi
  // rather than s_n, which is slightly less efficient.
  // ell'(x^T theta + at x^T grad(penalty) + ksi ||x||^2)
  double scale_factor(double ksi, double at, const data_point& data_pt, const
    mat& theta_old, double normx) const;
  // d/d(ksi) ell'
  double scale_factor_first_deriv(double ksi, double at, const data_point&
    data_pt, const mat& theta_old, double normx) const;
  // d^2/d(ksi)^2 ell'
  double scale_factor_second_deriv(double ksi, double at, const data_point&
    data_pt, const mat& theta_old, double normx) const;

protected:
  std::string name_;
  double lambda1_;
  double lambda2_;
};

#endif
