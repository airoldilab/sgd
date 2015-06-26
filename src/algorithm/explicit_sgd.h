#ifndef ALGORITHM_IMPLICIT_SGD_H
#define ALGORITHM_IMPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "learn-rate/learn_rate_value.h"
#include <stdlib.h>

template<typename EXPERIMENT>
mat explicit_sgd(unsigned t, const mat& theta_old, const data_set& data,
  EXPERIMENT& experiment, bool& good_gradient) {
  /* Return the new estimate of parameters, using SGD */
  data_point data_pt = data.get_data_point(t);
  unsigned idx = data.idxmap[t-1];
  learn_rate_value at = experiment.learning_rate(theta_old, data_pt,
    experiment.offset[idx], t);
  mat grad_t = experiment.gradient(theta_old, data_pt, experiment.offset[idx]);
  if (!is_finite(grad_t)) {
    good_gradient = false;
  }
  mat theta_new = theta_old + (at * grad_t);

  // Check the correctness of SGD update in DEBUG mode.
#if DEBUG
  if (!(at < 1)) {
    Rcpp::Rcout << "learning rate larger than 1 " <<
      "at Iter: " << t << std::endl;
  }
  mat theta_test;
  if (experiment.model_name == "gaussian" || experiment.model_name == "poisson"
    || experiment.model_name == "binomial" || experiment.model_name == "gamma") {
    theta_test = theta_old + at * ((data_pt.y - experiment.h_transfer(
      dot(data_pt.x, theta_old) + experiment.offset[idx]))*data_pt.x).t();
  } else{
    theta_test = theta_new;
  }
  double error = max(max(abs(theta_test - theta_new)));
  double scale = max(max(abs(theta_test)));
  if (error/scale > 1e-5) {
    Rcpp::Rcout<< "Wrong SGD update at iter: " << t + 1 << std::endl;
    Rcpp::Rcout<< "Relative Error = " <<  max(max(abs(theta_test - theta_new))) << std::endl;
    Rcpp::Rcout<< "Correct = " << theta_test << std::endl;
    Rcpp::Rcout<< "Output = " << theta_new << std::endl;
  }
#endif

  return theta_new;
}

#endif
