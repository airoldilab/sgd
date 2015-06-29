#ifndef ALGORITHM_EXPLICIT_SGD_H
#define ALGORITHM_EXPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "learn-rate/learn_rate_value.h"
#include <stdlib.h>

/**
 * Stochastic gradient descent (using an "explicit" update)
 *
 * @param  t             iteration
 * @param  theta_old     previous estimate
 * @param  data          data set
 * @tparam MODEL         model class
 * @param  good_gradient flag to store if gradient was computed okay
 */
template<typename MODEL>
mat explicit_sgd(unsigned t, const mat& theta_old, const data_set& data,
  MODEL& model, bool& good_gradient) {
  data_point data_pt = data.get_data_point(t);
  learn_rate_value at = model.learning_rate(theta_old, data_pt, t);
  mat grad_t = model.gradient(theta_old, data_pt);
  if (!is_finite(grad_t)) {
    good_gradient = false;
  }
  return theta_old + (at * grad_t);
}

#endif