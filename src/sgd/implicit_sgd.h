#ifndef SGD_IMPLICIT_SGD_H
#define SGD_IMPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "model/cox_model.h"
#include "model/ee_model.h"
#include "model/glm_model.h"
#include "learn-rate/learn_rate_value.h"
#include "sgd/base_sgd.h"

class implicit_sgd : public base_sgd {
  /**
   * Stochastic gradient descent using an "implicit" update
   *
   * @param sgd       attributes affiliated with sgd as R type
   * @param n_samples number of data samples
   * @param ti        timer for benchmarking how long to get each estimate
   */
public:
  implicit_sgd(Rcpp::List sgd, unsigned n_samples, const boost::timer& ti) :
    base_sgd(sgd, n_samples, ti) {
    delta_ = Rcpp::as<double>(sgd["delta"]);
  }

  mat update(unsigned t, const mat& theta_old, const data_set& data,
    glm_model& model, bool& good_gradient) {
    mat theta_new;
    learn_rate_value at = learning_rate(t, model.gradient(t, theta_old, data));
    // TODO how to deal with non-scalar learning rates?
    double at_avg = at.mean();

    data_point data_pt = data.get_data_point(t);
    double normx = dot(data_pt.x, data_pt.x);

    Get_grad_coeff<glm_model> get_grad_coeff(model, data_pt, theta_old,
      normx);
    Implicit_fn<glm_model> implicit_fn(at_avg, get_grad_coeff);

    double rt = at_avg * get_grad_coeff(0);
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

  mat update(unsigned t, const mat& theta_old, const data_set& data,
    cox_model& model, bool& good_gradient) {
    data_point data_pt = data.get_data_point(t);
    unsigned j = data_pt.idx;

    // assuming data points fail in order, i.e., risk set R_i={i,i+1,...,n}
    vec xi = exp(data.X * theta_old);
    vec h = zeros<vec>(j);
    double sum_xi = 0;
    for (int i = j-1; i < j; --i) {
      // h_i = d_i/sum(xi[i:n])
      if (i == j-1) {
        for (int k = i; k < data.n_samples; ++k) {
          sum_xi += xi(k);
        }
      } else {
        sum_xi += xi(i);
      }
      h(i) = data.Y(i)/sum_xi;
    }
    double eta_j = accu(data_pt.x.t() % theta_old); // x_j^T * theta
    double z = eta_j + data_pt.y - xi[j] * sum(h);
    double xjnorm = accu(data_pt.x % data_pt.x); // |x_j|^2_2

    //learn_rate_value at = learning_rate(t, model.gradient(t, theta_old, data));
    learn_rate_value at = learning_rate(t, zeros<mat>(data.n_features));
    // TODO how to deal with non-scalar learning rates?
    double at_avg = at.mean();

    mat grad_t = (z - (eta_j + at_avg*z*xjnorm)/(1 + at_avg*xjnorm)) *
      data_pt.x.t();
    if (!is_finite(grad_t)) {
      good_gradient = false;
    }
    return theta_old + (at * grad_t);
  }

  template <typename MODEL>
  mat update(unsigned t, const mat& theta_old, const data_set& data,
    MODEL& model, bool& good_gradient) {
    Rcpp::Rcout << "error: implicit not implemented for model yet" << std::endl;
    good_gradient = false;
    return theta_old;
  }

  implicit_sgd& operator=(const mat& theta_new) {
    base_sgd::operator=(theta_new);
    return *this;
  }
private:
  double delta_;
};

#endif
