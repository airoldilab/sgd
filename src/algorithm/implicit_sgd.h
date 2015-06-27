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
  mat theta_new;
  learn_rate_value at = experiment.learning_rate(theta_old, data_pt, t);
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
    normx);
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
  return theta_old + result * data_pt.x.t();
}

mat implicit_sgd(unsigned t, const mat& theta_old, const data_set& data,
  ee_experiment& experiment, bool& good_gradient) {
  //TODO
  Rcpp::Rcout << "error: implicit not implemented for EE yet " << t << std::endl;
  good_gradient = false;
  return theta_old;
}

#endif
