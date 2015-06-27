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
  learn_rate_value at = experiment.learning_rate(theta_old, data_pt, t);
  mat grad_t = experiment.gradient(theta_old, data_pt);
  if (!is_finite(grad_t)) {
    good_gradient = false;
  }
  return theta_old + (at * grad_t);
}

#endif
