#ifndef SGD_IMPLICIT_SGD_H
#define SGD_IMPLICIT_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "model/cox_model.h"
#include "model/glm_model.h"
#include "model/gmm_model.h"
#include "model/m_model.h"
#include "learn-rate/learn_rate_value.h"
#include "sgd/base_sgd.h"

template<typename MODEL>
class Implicit_fn {
  // Root finding functor for implicit update
  // Evaluates the zeroth, first, and second derivatives of:
  // ksi - ell(x^T theta + ||x||^2 * ksi)
public:
  typedef boost::math::tuple<double, double, double> tuple_type;

  Implicit_fn(const MODEL& m, double a, const data_point& d, const mat& t,
    double n) : model_(m), at_(a), data_pt_(d), theta_old_(t), normx_(n) {}

  tuple_type operator()(double ksi) const {
    double value = ksi - at_ *
      model_.scale_factor(ksi, at_, data_pt_, theta_old_, normx_);
    double first = 1 + at_ *
      model_.scale_factor_first_deriv(ksi, at_, data_pt_, theta_old_, normx_);
    double second = at_ *
      model_.scale_factor_second_deriv(ksi, at_, data_pt_, theta_old_, normx_);
    tuple_type out(value, first, second);
    return out;
  }

private:
  const MODEL& model_;
  double at_;
  const data_point& data_pt_;
  const mat& theta_old_;
  double normx_;
};

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

    double r = at_avg * model.scale_factor(0, at_avg, data_pt, theta_old, normx);
    double lower = 0;
    double upper = 0;
    if (r < 0) {
      lower = r;
    } else {
      upper = r;
    }
    double ksi;
    if (lower != upper) {
      Implicit_fn<glm_model> implicit_fn(model, at_avg, data_pt, theta_old, normx);
      ksi = boost::math::tools::schroeder_iterate(implicit_fn, (lower +
        upper)/2, lower, upper, delta_);
    } else {
      ksi = lower;
    }
    return theta_old +
      ksi * data_pt.x.t() -
      at_avg * model.gradient_penalty(theta_old);
  }

  mat update(unsigned t, const mat& theta_old, const data_set& data,
    m_model& model, bool& good_gradient) {
    mat theta_new;
    learn_rate_value at = learning_rate(t, model.gradient(t, theta_old, data));
    // TODO how to deal with non-scalar learning rates?
    double at_avg = at.mean();

    data_point data_pt = data.get_data_point(t);
    double normx = dot(data_pt.x, data_pt.x);

    double r = at_avg * model.scale_factor(0, at_avg, data_pt, theta_old, normx);
    double lower = 0;
    double upper = 0;
    if (r < 0) {
      lower = r;
    } else {
      upper = r;
    }
    double ksi;
    if (lower != upper) {
      Implicit_fn<m_model> implicit_fn(model, at_avg, data_pt, theta_old, normx);
      ksi = boost::math::tools::schroeder_iterate(implicit_fn, (lower +
        upper)/2, lower, upper, delta_);
    } else {
      ksi = lower;
    }
    return theta_old +
      ksi * data_pt.x.t() -
      at_avg * model.gradient_penalty(theta_old);
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
