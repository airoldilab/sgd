#ifndef ALGORITHM_EXPLICIT_SGD_H
#define ALGORITHM_EXPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "experiment/ee_experiment.h"
#include "experiment/glm_experiment.h"
#include "learn-rate/learn_rate_value.h"
#include <stdlib.h>

mat implicit_sgd(unsigned t, const mat& theta_old, const data_set& data,
  glm_experiment& experiment, bool& good_gradient) {
  /* return the new estimate of parameters, using implicit SGD */
  data_point data_pt = data.get_data_point(t);
  unsigned idx = data.idxmap[t-1];
  mat theta_new;
  learn_rate_value at = experiment.learning_rate(theta_old, data_pt, experiment.offset[idx], t);
  double average_lr = 0;
  if (at.type == 0) {
    average_lr = at.lr_scalar;
  } else {
    vec diag_lr = at.lr_mat.diag();
    for (unsigned i = 0; i < diag_lr.n_elem; ++i) {
      average_lr += diag_lr[i];
    }
    average_lr /= diag_lr.n_elem;
  }

  double normx = dot(data_pt.x, data_pt.x);

  Get_grad_coeff<glm_experiment> get_grad_coeff(experiment, data_pt, theta_old,
    normx, experiment.offset[idx]);
  Implicit_fn<glm_experiment> implicit_fn(average_lr, get_grad_coeff);

  double rt = average_lr * get_grad_coeff(0);
  double lower = 0;
  double upper = 0;
  if (rt < 0) {
    upper = 0;
    lower = rt;
  } else {
    // double u = 0;
    // u = (experiment.g_link(data_pt.y) - dot(theta_old,data_pt.x))/normx;
    // upper = std::min(rt, u);
    // lower = 0;
    upper = rt;
    lower = 0;
  }
  double result;
  if (lower != upper) {
    result = boost::math::tools::schroeder_iterate(implicit_fn, (lower +
      upper)/2, lower, upper, experiment.delta);
  } else {
    result = lower;
  }
  theta_new = theta_old + result * data_pt.x.t();

  // Check the correctness of SGD update in DEBUG mode.
#if DEBUG
  if (!(average_lr < 1)) {
    Rcpp::Rcout << "learning rate larger than 1" <<
      "at Iter: " << t << std::endl;
    Rcpp::Rcout << "lr = " << average_lr <<std::endl;
  }
  mat theta_test;
  if (experiment.model_name == "gaussian" ||
      experiment.model_name == "poisson" ||
      experiment.model_name == "binomial" ||
      experiment.model_name == "gamma") {
    theta_test = theta_new - average_lr * ((data_pt.y - experiment.h_transfer(
      dot(data_pt.x, theta_new) + experiment.offset[idx]))*data_pt.x).t();
  } else {
    theta_test = theta_old;
  }
  double error = max(max(abs(theta_test - theta_old)));
  double scale = max(max(abs(theta_test)));
  if (error/scale > 1e-5 && error > 1e-5) {
    Rcpp::Rcout<< "Wrong SGD update at iter: " << t + 1 << std::endl;
    Rcpp::Rcout<< "Max Error = " <<  max(max(abs(theta_test - theta_old))) << std::endl;
    Rcpp::Rcout<< "test = " << theta_test << std::endl;
    Rcpp::Rcout<< "new = " << theta_new << std::endl;
    Rcpp::Rcout<< "old = " << theta_old << std::endl;
    Rcpp::Rcout<< "result = " << result << std::endl;
    Rcpp::Rcout<< "f(result) = " << implicit_fn(result) <<std::endl;
    Rcpp::Rcout<< "lr = " << average_lr <<std::endl;
    Rcpp::Rcout<< "data.x = " << data_pt.x <<std::endl;
    Rcpp::Rcout<< "data.y = " << data_pt.y <<std::endl;
    Rcpp::Rcout<< "normx = " << normx <<std::endl;
  }
#endif

  return theta_new;
}

mat implicit_sgd(unsigned t, const mat& theta_old, const data_set& data,
  ee_experiment& experiment, bool& good_gradient) {
  //TODO
  Rcpp::Rcout << "error: implicit not implemented for EE yet " << t << std::endl;
  good_gradient = false;
  return theta_old;
}

#endif
