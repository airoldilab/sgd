#ifndef SGD_IMPLICIT_SGD_H
#define SGD_IMPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "model/ee_model.h"
#include "model/glm_model.h"
#include "learn-rate/learn_rate_value.h"
#include "sgd/base_sgd.h"
#include <stdlib.h>

class implicit_sgd : public base_sgd {
  /**
   * Stochastic gradient descent (using an "implicit" update)
   *
   * @param sgd       attributes affiliated with sgd as R type
   * @param n_samples number of data samples
   * @param ti        timer for benchmarking how long to get each estimate
   */
public:
  implicit_sgd(Rcpp::List sgd, unsigned n_samples, const boost::timer& ti,
    grad_func_type grad_func) : base_sgd(sgd, n_samples, ti, grad_func) {}

  mat update(unsigned t, const mat& theta_old, const data_set& data,
    glm_model& model, bool& good_gradient) {
    data_point data_pt = data.get_data_point(t);
    mat theta_new;
    learn_rate_value at = learning_rate(theta_old, data_pt, t);
    // TODO
    double average_lr = at.mean();

    double normx = dot(data_pt.x, data_pt.x);

    Get_grad_coeff<glm_model> get_grad_coeff(model, data_pt, theta_old,
      normx);
    Implicit_fn<glm_model> implicit_fn(average_lr, get_grad_coeff);

    double rt = average_lr * get_grad_coeff(0);
    double lower = 0;
    double upper = 0;
    if (rt < 0) {
      upper = 0;
      lower = rt;
    } else {
      // double u = 0;
      // u = (model.g_link(data_pt.y) - dot(theta_old,data_pt.x))/normx;
      // upper = std::min(rt, u);
      // lower = 0;
      upper = rt;
      lower = 0;
    }
    double result;
    if (lower != upper) {
      result = boost::math::tools::schroeder_iterate(implicit_fn, (lower +
        upper)/2, lower, upper, delta_);
    } else {
      result = lower;
    }
    return theta_old + result * data_pt.x.t();
  }

  template <typename MODEL>
  mat update(unsigned t, const mat& theta_old, const data_set& data,
    MODEL& model, bool& good_gradient) {
    Rcpp::Rcout << "error: implicit not implemented for model yet" << std::endl;
    good_gradient = false;
    return theta_old;
  }

  // Operators
  implicit_sgd& operator=(const mat& theta_new) {
    base_sgd::operator=(theta_new);
    return *this;
  }
};

#endif
