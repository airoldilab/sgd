#ifndef VALIDITY_CHECK_VALIDITY_CHECK_H
#define VALIDITY_CHECK_VALIDITY_CHECK_H

#include "basedef.h"
#include "data/data_set.h"
#include "validity-check/cox_validity_check_model.h"
#include "validity-check/glm_validity_check_model.h"
#include "validity-check/gmm_validity_check_model.h"

template<typename MODEL>
bool validity_check(const data_set& data, const mat& theta, bool good_gradient,
  unsigned t, const MODEL& model) {
  // Check if gradient is finite.
  if (!good_gradient) {
    Rcpp::Rcout << "error: NA or infinite gradient" << std::endl;
    return false;
  }

  // Check if all estimates are finite.
  if (!is_finite(theta)) {
    Rcpp::Rcout << "warning: non-finite coefficients at iteration " << t << std::endl;
  }

  return validity_check_model(data, theta, t, model);
}

#endif
