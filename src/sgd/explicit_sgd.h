#ifndef SGD_EXPLICIT_SGD_H
#define SGD_EXPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "learn-rate/learn_rate_value.h"
#include "sgd/base_sgd.h"
#include <stdlib.h>

class explicit_sgd : public base_sgd {
  /**
   * Stochastic gradient descent in standard formulation, i.e., using an
   * "explicit" update
   *
   * @param sgd       attributes affiliated with sgd as R type
   * @param n_samples number of data samples
   * @param ti        timer for benchmarking how long to get each estimate
   */
public:
  explicit_sgd(Rcpp::List sgd, unsigned n_samples, const boost::timer& ti) :
    base_sgd(sgd, n_samples, ti) {}

  template<typename MODEL>
  mat update(unsigned t, const mat& theta_old, const data_set& data,
    MODEL& model, bool& good_gradient) {
    data_point data_pt = data.get_data_point(t);
    mat grad_t = model.gradient(theta_old, data_pt, data);
    learn_rate_value at = learning_rate(grad_t, t);
    if (!is_finite(grad_t)) {
      good_gradient = false;
    }
    return theta_old + (at * grad_t);
  }

  explicit_sgd& operator=(const mat& theta_new) {
    base_sgd::operator=(theta_new);
    return *this;
  }
};

#endif
